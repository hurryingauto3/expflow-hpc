"""
Results Storage System for ExpFlow v0.8.0

Provides unified database backend for experiment results with support for:
- SQLite (default, file-based, no dependencies)
- MongoDB (scalable, optional dependency)
- PostgreSQL (production, optional dependency)

Design Philosophy:
- Abstract backend pattern for flexibility
- SQLite as default (zero setup)
- Rich query API for filtering and analysis
- Export utilities for web visualization
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager


class BaseDatabaseBackend(ABC):
    """Abstract base class for database backends"""

    @abstractmethod
    def connect(self):
        """Establish database connection"""
        pass

    @abstractmethod
    def disconnect(self):
        """Close database connection"""
        pass

    @abstractmethod
    def store_experiment(self, exp_id: str, data: Dict[str, Any]) -> bool:
        """Store experiment data"""
        pass

    @abstractmethod
    def update_experiment(self, exp_id: str, data: Dict[str, Any]) -> bool:
        """Update existing experiment data"""
        pass

    @abstractmethod
    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single experiment by ID"""
        pass

    @abstractmethod
    def query_experiments(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query experiments with optional filters"""
        pass

    @abstractmethod
    def delete_experiment(self, exp_id: str) -> bool:
        """Delete experiment from database"""
        pass

    @abstractmethod
    def count_experiments(self, filters: Dict[str, Any] = None) -> int:
        """Count experiments matching filters"""
        pass


class SQLiteBackend(BaseDatabaseBackend):
    """SQLite database backend (default, zero dependencies)"""

    def __init__(self, db_path: str = None):
        """
        Initialize SQLite backend

        Args:
            db_path: Path to SQLite database file (default: experiments.db in current dir)
        """
        self.db_path = db_path or "experiments.db"
        self.conn = None
        self._create_tables()

    def connect(self):
        """Establish SQLite connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Return dict-like rows

    def disconnect(self):
        """Close SQLite connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _create_tables(self):
        """Create experiments table if not exists"""
        self.connect()
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                exp_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on common query fields
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON experiments(created_at)
        """)

        self.conn.commit()

    def store_experiment(self, exp_id: str, data: Dict[str, Any]) -> bool:
        """
        Store experiment data

        Args:
            exp_id: Unique experiment identifier
            data: Experiment data dictionary (will be stored as JSON)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.connect()

            # Add exp_id to data if not present
            if 'exp_id' not in data:
                data['exp_id'] = exp_id

            # Serialize data to JSON
            json_data = json.dumps(data)

            self.conn.execute("""
                INSERT OR REPLACE INTO experiments (exp_id, data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (exp_id, json_data))

            self.conn.commit()
            return True
        except Exception as e:
            print(f"ERROR storing experiment {exp_id}: {e}")
            return False

    def update_experiment(self, exp_id: str, data: Dict[str, Any]) -> bool:
        """
        Update existing experiment data

        Args:
            exp_id: Experiment identifier
            data: New/updated data fields (merged with existing)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.connect()

            # Get existing data
            existing = self.get_experiment(exp_id)
            if not existing:
                # If doesn't exist, create new
                return self.store_experiment(exp_id, data)

            # Merge with new data
            existing.update(data)

            # Store updated data
            json_data = json.dumps(existing)
            self.conn.execute("""
                UPDATE experiments
                SET data = ?, updated_at = CURRENT_TIMESTAMP
                WHERE exp_id = ?
            """, (json_data, exp_id))

            self.conn.commit()
            return True
        except Exception as e:
            print(f"ERROR updating experiment {exp_id}: {e}")
            return False

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve single experiment by ID

        Args:
            exp_id: Experiment identifier

        Returns:
            Experiment data dictionary or None if not found
        """
        try:
            self.connect()
            cursor = self.conn.execute(
                "SELECT data FROM experiments WHERE exp_id = ?",
                (exp_id,)
            )
            row = cursor.fetchone()

            if row:
                return json.loads(row['data'])
            return None
        except Exception as e:
            print(f"ERROR retrieving experiment {exp_id}: {e}")
            return None

    def query_experiments(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query experiments with optional filters

        Args:
            filters: Dictionary of field filters (supports nested JSON queries)
                    Example: {'status': 'completed', 'metrics.accuracy': '>0.9'}

        Returns:
            List of experiment data dictionaries
        """
        try:
            self.connect()

            if not filters:
                # Return all experiments
                cursor = self.conn.execute(
                    "SELECT data FROM experiments ORDER BY created_at DESC"
                )
            else:
                # Build WHERE clause for JSON queries
                # This is a simplified implementation
                # For complex queries, consider using json_extract in SQLite 3.38+
                cursor = self.conn.execute(
                    "SELECT data FROM experiments ORDER BY created_at DESC"
                )

            results = []
            for row in cursor:
                exp_data = json.loads(row['data'])

                # Apply filters in Python (post-query filtering)
                if filters and not self._matches_filters(exp_data, filters):
                    continue

                results.append(exp_data)

            return results
        except Exception as e:
            print(f"ERROR querying experiments: {e}")
            return []

    def _matches_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if experiment data matches filters

        Supports nested field access with dot notation: 'metrics.accuracy'
        Supports comparison operators: '>0.9', '>=0.9', '<0.1', '==completed'
        """
        for field, value in filters.items():
            # Get nested field value
            field_value = self._get_nested_value(data, field)

            if field_value is None:
                return False

            # Handle comparison operators
            if isinstance(value, str):
                if value.startswith('>='):
                    if not (field_value >= float(value[2:])):
                        return False
                elif value.startswith('>'):
                    if not (field_value > float(value[1:])):
                        return False
                elif value.startswith('<='):
                    if not (field_value <= float(value[2:])):
                        return False
                elif value.startswith('<'):
                    if not (field_value < float(value[1:])):
                        return False
                elif value.startswith('=='):
                    if field_value != value[2:]:
                        return False
                else:
                    # Exact match or substring match for strings
                    if isinstance(field_value, str):
                        if value.lower() not in field_value.lower():
                            return False
                    elif field_value != value:
                        return False
            else:
                # Direct comparison
                if field_value != value:
                    return False

        return True

    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """Get nested field value using dot notation"""
        keys = field.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def delete_experiment(self, exp_id: str) -> bool:
        """Delete experiment from database"""
        try:
            self.connect()
            self.conn.execute("DELETE FROM experiments WHERE exp_id = ?", (exp_id,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"ERROR deleting experiment {exp_id}: {e}")
            return False

    def count_experiments(self, filters: Dict[str, Any] = None) -> int:
        """Count experiments matching filters"""
        experiments = self.query_experiments(filters)
        return len(experiments)


class MongoDBBackend(BaseDatabaseBackend):
    """MongoDB database backend (requires pymongo)"""

    def __init__(self, connection_string: str = None, database: str = "expflow", collection: str = "experiments"):
        """
        Initialize MongoDB backend

        Args:
            connection_string: MongoDB connection string (default: mongodb://localhost:27017/)
            database: Database name (default: expflow)
            collection: Collection name (default: experiments)

        Example:
            # Local MongoDB
            backend = MongoDBBackend()

            # Remote MongoDB (MongoDB Atlas)
            backend = MongoDBBackend(
                connection_string="mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority",
                database="my_experiments"
            )
        """
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure
        except ImportError:
            raise ImportError(
                "MongoDB backend requires pymongo. Install with: pip install pymongo"
            )

        self.connection_string = connection_string or "mongodb://localhost:27017/"
        self.database_name = database
        self.collection_name = collection
        self.client = None
        self.db = None
        self.collection = None
        self.MongoClient = MongoClient
        self.ConnectionFailure = ConnectionFailure

    def connect(self):
        """Establish MongoDB connection"""
        if self.client is None:
            try:
                self.client = self.MongoClient(self.connection_string)
                # Test connection
                self.client.admin.command('ping')
                self.db = self.client[self.database_name]
                self.collection = self.db[self.collection_name]

                # Create indexes for common queries
                self.collection.create_index("exp_id", unique=True)
                self.collection.create_index("status")
                self.collection.create_index("created_at")
            except self.ConnectionFailure as e:
                print(f"ERROR connecting to MongoDB: {e}")
                raise

    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None

    def store_experiment(self, exp_id: str, data: Dict[str, Any]) -> bool:
        """Store experiment data"""
        try:
            self.connect()

            # Add exp_id to data if not present
            if 'exp_id' not in data:
                data['exp_id'] = exp_id

            # Add timestamps
            data['stored_at'] = datetime.now().isoformat()

            # Use upsert (insert or replace)
            self.collection.replace_one(
                {'exp_id': exp_id},
                data,
                upsert=True
            )
            return True
        except Exception as e:
            print(f"ERROR storing experiment {exp_id}: {e}")
            return False

    def update_experiment(self, exp_id: str, data: Dict[str, Any]) -> bool:
        """Update existing experiment data"""
        try:
            self.connect()

            # Get existing data
            existing = self.get_experiment(exp_id)
            if not existing:
                # If doesn't exist, create new
                return self.store_experiment(exp_id, data)

            # Merge with new data
            existing.update(data)
            existing['updated_at'] = datetime.now().isoformat()

            # Update in database
            self.collection.replace_one({'exp_id': exp_id}, existing)
            return True
        except Exception as e:
            print(f"ERROR updating experiment {exp_id}: {e}")
            return False

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single experiment by ID"""
        try:
            self.connect()
            doc = self.collection.find_one({'exp_id': exp_id})
            if doc:
                # Remove MongoDB's _id field
                doc.pop('_id', None)
                return doc
            return None
        except Exception as e:
            print(f"ERROR retrieving experiment {exp_id}: {e}")
            return None

    def query_experiments(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query experiments with optional filters

        MongoDB supports native nested field queries:
        filters = {'status': 'completed', 'results.pdm_score': {'$gt': 0.8}}
        """
        try:
            self.connect()

            # Convert ExpFlow filters to MongoDB query
            mongo_query = self._convert_filters_to_mongo(filters or {})

            # Query database
            cursor = self.collection.find(mongo_query).sort('created_at', -1)

            # Convert to list and remove MongoDB _id
            results = []
            for doc in cursor:
                doc.pop('_id', None)
                results.append(doc)

            return results
        except Exception as e:
            print(f"ERROR querying experiments: {e}")
            return []

    def _convert_filters_to_mongo(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ExpFlow filters to MongoDB query format"""
        mongo_query = {}

        for field, value in filters.items():
            if isinstance(value, str):
                # Handle comparison operators
                if value.startswith('>='):
                    mongo_query[field] = {'$gte': float(value[2:])}
                elif value.startswith('>'):
                    mongo_query[field] = {'$gt': float(value[1:])}
                elif value.startswith('<='):
                    mongo_query[field] = {'$lte': float(value[2:])}
                elif value.startswith('<'):
                    mongo_query[field] = {'$lt': float(value[1:])}
                elif value.startswith('=='):
                    mongo_query[field] = value[2:]
                else:
                    # Regex match for strings
                    mongo_query[field] = {'$regex': value, '$options': 'i'}
            else:
                # Direct match
                mongo_query[field] = value

        return mongo_query

    def delete_experiment(self, exp_id: str) -> bool:
        """Delete experiment from database"""
        try:
            self.connect()
            result = self.collection.delete_one({'exp_id': exp_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"ERROR deleting experiment {exp_id}: {e}")
            return False

    def count_experiments(self, filters: Dict[str, Any] = None) -> int:
        """Count experiments matching filters"""
        try:
            self.connect()
            mongo_query = self._convert_filters_to_mongo(filters or {})
            return self.collection.count_documents(mongo_query)
        except Exception as e:
            print(f"ERROR counting experiments: {e}")
            return 0


class PostgreSQLBackend(BaseDatabaseBackend):
    """PostgreSQL database backend (requires psycopg2)"""

    def __init__(self, connection_string: str = None, table_name: str = "experiments"):
        """
        Initialize PostgreSQL backend

        Args:
            connection_string: PostgreSQL connection string
            table_name: Table name (default: experiments)

        Example:
            # Local PostgreSQL
            backend = PostgreSQLBackend(
                connection_string="postgresql://user:password@localhost:5432/expflow"
            )

            # Remote PostgreSQL (e.g., Amazon RDS, Google Cloud SQL)
            backend = PostgreSQLBackend(
                connection_string="postgresql://user:pass@host.amazonaws.com:5432/expflow"
            )
        """
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError(
                "PostgreSQL backend requires psycopg2. Install with: pip install psycopg2-binary"
            )

        if not connection_string:
            raise ValueError("PostgreSQL backend requires connection_string parameter")

        self.connection_string = connection_string
        self.table_name = table_name
        self.conn = None
        self.psycopg2 = psycopg2

    def connect(self):
        """Establish PostgreSQL connection"""
        if self.conn is None:
            try:
                self.conn = self.psycopg2.connect(self.connection_string)
                self._create_tables()
            except Exception as e:
                print(f"ERROR connecting to PostgreSQL: {e}")
                raise

    def disconnect(self):
        """Close PostgreSQL connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _create_tables(self):
        """Create experiments table if not exists"""
        with self.conn.cursor() as cur:
            # Create table with JSONB for flexible schema
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    exp_id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for common queries (JSONB supports indexing)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_status
                ON {self.table_name} ((data->>'status'))
            """)

            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created_at
                ON {self.table_name} (created_at)
            """)

            # GIN index for JSONB queries
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_data_gin
                ON {self.table_name} USING GIN (data)
            """)

            self.conn.commit()

    def store_experiment(self, exp_id: str, data: Dict[str, Any]) -> bool:
        """Store experiment data"""
        try:
            self.connect()

            # Add exp_id to data if not present
            if 'exp_id' not in data:
                data['exp_id'] = exp_id

            # Serialize data to JSON
            json_data = json.dumps(data)

            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table_name} (exp_id, data, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (exp_id)
                    DO UPDATE SET data = EXCLUDED.data, updated_at = CURRENT_TIMESTAMP
                """, (exp_id, json_data))

            self.conn.commit()
            return True
        except Exception as e:
            print(f"ERROR storing experiment {exp_id}: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def update_experiment(self, exp_id: str, data: Dict[str, Any]) -> bool:
        """Update existing experiment data"""
        try:
            self.connect()

            # Get existing data
            existing = self.get_experiment(exp_id)
            if not existing:
                # If doesn't exist, create new
                return self.store_experiment(exp_id, data)

            # Merge with new data
            existing.update(data)

            # Store updated data
            json_data = json.dumps(existing)

            with self.conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {self.table_name}
                    SET data = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE exp_id = %s
                """, (json_data, exp_id))

            self.conn.commit()
            return True
        except Exception as e:
            print(f"ERROR updating experiment {exp_id}: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single experiment by ID"""
        try:
            self.connect()

            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT data FROM {self.table_name} WHERE exp_id = %s
                """, (exp_id,))

                row = cur.fetchone()
                if row:
                    return json.loads(row[0]) if isinstance(row[0], str) else row[0]

            return None
        except Exception as e:
            print(f"ERROR retrieving experiment {exp_id}: {e}")
            return None

    def query_experiments(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query experiments with optional filters

        PostgreSQL supports JSONB queries:
        filters = {'status': 'completed', 'results.pdm_score': '>0.8'}
        """
        try:
            self.connect()

            # Build WHERE clause for JSONB queries
            where_clauses = []
            params = []

            if filters:
                for field, value in filters.items():
                    # Convert dot notation to JSONB path
                    json_path = "->".join([f"'{part}'" for part in field.split('.')])

                    if isinstance(value, str):
                        # Handle comparison operators
                        if value.startswith('>='):
                            where_clauses.append(f"(data->{json_path})::float >= %s")
                            params.append(float(value[2:]))
                        elif value.startswith('>'):
                            where_clauses.append(f"(data->{json_path})::float > %s")
                            params.append(float(value[1:]))
                        elif value.startswith('<='):
                            where_clauses.append(f"(data->{json_path})::float <= %s")
                            params.append(float(value[2:]))
                        elif value.startswith('<'):
                            where_clauses.append(f"(data->{json_path})::float < %s")
                            params.append(float(value[1:]))
                        elif value.startswith('=='):
                            where_clauses.append(f"data->{json_path} = %s")
                            params.append(json.dumps(value[2:]))
                        else:
                            # String contains (case-insensitive)
                            where_clauses.append(f"data->{json_path} ILIKE %s")
                            params.append(f"%{value}%")
                    else:
                        # Direct match
                        where_clauses.append(f"data->{json_path} = %s")
                        params.append(json.dumps(value))

            # Build query
            query = f"SELECT data FROM {self.table_name}"
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            query += " ORDER BY created_at DESC"

            with self.conn.cursor() as cur:
                cur.execute(query, params)
                results = []
                for row in cur.fetchall():
                    data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    results.append(data)

            return results
        except Exception as e:
            print(f"ERROR querying experiments: {e}")
            return []

    def delete_experiment(self, exp_id: str) -> bool:
        """Delete experiment from database"""
        try:
            self.connect()

            with self.conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_name} WHERE exp_id = %s", (exp_id,))

            self.conn.commit()
            return True
        except Exception as e:
            print(f"ERROR deleting experiment {exp_id}: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def count_experiments(self, filters: Dict[str, Any] = None) -> int:
        """Count experiments matching filters"""
        experiments = self.query_experiments(filters)
        return len(experiments)


class ResultsStorage:
    """
    High-level interface for experiment results storage

    Provides context manager support and backend abstraction
    """

    def __init__(self, backend: str = 'sqlite', **backend_kwargs):
        """
        Initialize results storage

        Args:
            backend: Backend type ('sqlite', 'mongodb', 'postgresql')
            **backend_kwargs: Backend-specific configuration

        Example:
            # SQLite (default, local)
            storage = ResultsStorage(backend='sqlite', path='experiments.db')

            # MongoDB (remote, requires pymongo)
            storage = ResultsStorage(
                backend='mongodb',
                connection_string='mongodb+srv://user:pass@cluster.mongodb.net/',
                database='my_experiments'
            )

            # PostgreSQL (remote, requires psycopg2-binary)
            storage = ResultsStorage(
                backend='postgresql',
                connection_string='postgresql://user:pass@host:5432/expflow'
            )
        """
        self.backend_type = backend

        if backend == 'sqlite':
            db_path = backend_kwargs.get('path', 'experiments.db')
            self.backend = SQLiteBackend(db_path)
        elif backend == 'mongodb':
            connection_string = backend_kwargs.get('connection_string')
            database = backend_kwargs.get('database', 'expflow')
            collection = backend_kwargs.get('collection', 'experiments')
            self.backend = MongoDBBackend(connection_string, database, collection)
        elif backend == 'postgresql':
            connection_string = backend_kwargs.get('connection_string')
            table_name = backend_kwargs.get('table_name', 'experiments')
            self.backend = PostgreSQLBackend(connection_string, table_name)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def __enter__(self):
        """Context manager entry"""
        self.backend.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.backend.disconnect()

    def store(self, exp_id: str, data: Dict[str, Any]) -> bool:
        """Store experiment data"""
        return self.backend.store_experiment(exp_id, data)

    def update(self, exp_id: str, data: Dict[str, Any]) -> bool:
        """Update experiment data"""
        return self.backend.update_experiment(exp_id, data)

    def get(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Get single experiment"""
        return self.backend.get_experiment(exp_id)

    def query(self, **filters) -> List[Dict[str, Any]]:
        """Query experiments with filters"""
        return self.backend.query_experiments(filters if filters else None)

    def delete(self, exp_id: str) -> bool:
        """Delete experiment"""
        return self.backend.delete_experiment(exp_id)

    def count(self, **filters) -> int:
        """Count experiments"""
        return self.backend.count_experiments(filters if filters else None)


class ResultsQueryAPI:
    """
    High-level query API for experiment analysis

    Provides convenience methods for common queries
    """

    def __init__(self, storage: ResultsStorage):
        """
        Initialize query API

        Args:
            storage: ResultsStorage instance
        """
        self.storage = storage

    def best_experiments(
        self,
        metric: str,
        n: int = 10,
        ascending: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get top N experiments by metric

        Args:
            metric: Metric field (supports dot notation: 'metrics.accuracy')
            n: Number of results
            ascending: Sort order (True=lowest first, False=highest first)

        Returns:
            List of top experiments

        Example:
            # Get top 10 experiments by PDM score
            best = api.best_experiments('evaluation.navtest.pdm_score', n=10, ascending=False)
        """
        all_experiments = self.storage.query()

        # Extract metric values
        experiments_with_metric = []
        for exp in all_experiments:
            metric_value = self._get_nested_value(exp, metric)
            if metric_value is not None:
                experiments_with_metric.append((exp, metric_value))

        # Sort by metric
        experiments_with_metric.sort(key=lambda x: x[1], reverse=not ascending)

        # Return top N
        return [exp for exp, _ in experiments_with_metric[:n]]

    def search(
        self,
        status: str = None,
        partition: str = None,
        min_metric: Dict[str, float] = None,
        max_metric: Dict[str, float] = None,
        created_after: str = None,
        created_before: str = None
    ) -> List[Dict[str, Any]]:
        """
        Advanced search with multiple filters

        Args:
            status: Filter by status ('completed', 'running', 'failed')
            partition: Filter by SLURM partition
            min_metric: Minimum values for metrics {'pdm_score': 0.8, 'accuracy': 0.9}
            max_metric: Maximum values for metrics
            created_after: Filter by creation date (ISO format)
            created_before: Filter by creation date (ISO format)

        Returns:
            List of matching experiments

        Example:
            # Find completed experiments on L40s with PDM > 0.8
            results = api.search(
                status='completed',
                partition='l40s_public',
                min_metric={'evaluation.navtest.pdm_score': 0.8}
            )
        """
        filters = {}

        if status:
            filters['status'] = status

        if partition:
            filters['slurm.partition'] = partition

        # Apply filters
        results = self.storage.query(**filters)

        # Post-filter for metrics
        if min_metric:
            results = [
                exp for exp in results
                if all(
                    self._get_nested_value(exp, key) is not None and
                    self._get_nested_value(exp, key) >= value
                    for key, value in min_metric.items()
                )
            ]

        if max_metric:
            results = [
                exp for exp in results
                if all(
                    self._get_nested_value(exp, key) is not None and
                    self._get_nested_value(exp, key) <= value
                    for key, value in max_metric.items()
                )
            ]

        # Date filtering
        if created_after or created_before:
            results = self._filter_by_date(results, created_after, created_before)

        return results

    def compare_experiments(self, exp_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple experiments side-by-side

        Args:
            exp_ids: List of experiment IDs to compare

        Returns:
            Dictionary mapping exp_id to experiment data
        """
        comparison = {}
        for exp_id in exp_ids:
            exp = self.storage.get(exp_id)
            if exp:
                comparison[exp_id] = exp

        return comparison

    def get_statistics(self, metric: str) -> Dict[str, float]:
        """
        Calculate statistics for a metric across all experiments

        Args:
            metric: Metric field (supports dot notation)

        Returns:
            Dictionary with min, max, mean, median, std
        """
        all_experiments = self.storage.query()

        values = []
        for exp in all_experiments:
            value = self._get_nested_value(exp, metric)
            if value is not None and isinstance(value, (int, float)):
                values.append(value)

        if not values:
            return {}

        import statistics
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0
        }

    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """Get nested field value using dot notation"""
        keys = field.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _filter_by_date(
        self,
        experiments: List[Dict[str, Any]],
        after: str = None,
        before: str = None
    ) -> List[Dict[str, Any]]:
        """Filter experiments by creation date"""
        from dateutil import parser

        filtered = experiments

        if after:
            after_date = parser.parse(after)
            filtered = [
                exp for exp in filtered
                if 'created_at' in exp and parser.parse(exp['created_at']) >= after_date
            ]

        if before:
            before_date = parser.parse(before)
            filtered = [
                exp for exp in filtered
                if 'created_at' in exp and parser.parse(exp['created_at']) <= before_date
            ]

        return filtered


# Export utility functions

def export_to_json(
    storage: ResultsStorage,
    output_path: str = 'experiments_export.json',
    filters: Dict[str, Any] = None
):
    """
    Export experiments to JSON file for web visualization

    Args:
        storage: ResultsStorage instance
        output_path: Output JSON file path
        filters: Optional filters to apply

    Example:
        with ResultsStorage(backend='sqlite', path='experiments.db') as storage:
            export_to_json(storage, 'public_results.json', filters={'status': 'completed'})
    """
    experiments = storage.query(**(filters or {}))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(experiments, f, indent=2)

    print(f"Exported {len(experiments)} experiments to {output_path}")


def export_to_csv(
    storage: ResultsStorage,
    output_path: str = 'experiments_export.csv',
    fields: List[str] = None,
    filters: Dict[str, Any] = None
):
    """
    Export experiments to CSV file

    Args:
        storage: ResultsStorage instance
        output_path: Output CSV file path
        fields: List of fields to export (supports dot notation)
        filters: Optional filters to apply

    Example:
        fields = ['exp_id', 'status', 'evaluation.navtest.pdm_score', 'metrics.val_loss']
        export_to_csv(storage, 'results.csv', fields=fields)
    """
    import csv

    experiments = storage.query(**(filters or {}))

    if not experiments:
        print("No experiments to export")
        return

    # Auto-detect fields if not provided
    if not fields:
        fields = ['exp_id', 'status', 'created_at']

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def get_nested(data, field):
        """Helper to get nested values"""
        keys = field.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for exp in experiments:
            row = {field: get_nested(exp, field) for field in fields}
            writer.writerow(row)

    print(f"Exported {len(experiments)} experiments to {output_path}")
