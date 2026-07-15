"""Reusable MySQL persistence helpers for the HDI Prediction System.

Database connection values are read from environment variables so credentials
are never committed to source control:

    HDI_DB_HOST, HDI_DB_PORT, HDI_DB_NAME, HDI_DB_USER, HDI_DB_PASSWORD

Run schema.sql once before using persistence. The Flask application continues
to serve ML predictions if MySQL is temporarily unavailable; the failed write
is logged and can be retried on the next prediction request.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional

try:
    import mysql.connector
    from mysql.connector import Error
    from mysql.connector.pooling import MySQLConnectionPool
except ImportError:  # Allows the Flask UI to run before dependencies are installed.
    mysql = None
    Error = Exception
    MySQLConnectionPool = None
else:
    mysql = mysql.connector


logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": os.getenv("HDI_DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("HDI_DB_PORT", "3306")),
    "database": os.getenv("HDI_DB_NAME", "hdi_prediction_db"),
    "user": os.getenv("HDI_DB_USER", "root"),
    "password": os.getenv("HDI_DB_PASSWORD", ""),
}
POOL_NAME = "hdi_prediction_pool"
POOL_SIZE = int(os.getenv("HDI_DB_POOL_SIZE", "5"))

_connection_pool: Optional[Any] = None
_pool_lock = Lock()


class DatabaseOperationError(RuntimeError):
    """Raised when a database operation cannot safely be completed."""


def _get_pool() -> Any:
    """Create one thread-safe MySQL connection pool and reuse it thereafter."""
    global _connection_pool
    if mysql is None or MySQLConnectionPool is None:
        raise DatabaseOperationError(
            "MySQL connector is not installed. Run `pip install -r requirements.txt`."
        )

    with _pool_lock:
        if _connection_pool is None:
            try:
                _connection_pool = MySQLConnectionPool(
                    pool_name=POOL_NAME,
                    pool_size=POOL_SIZE,
                    pool_reset_session=True,
                    **DB_CONFIG,
                )
            except Error as exc:
                raise DatabaseOperationError(
                    "Unable to create the MySQL connection pool. Check the database "
                    "service, credentials, and that schema.sql has been executed."
                ) from exc
    return _connection_pool


def get_connection() -> Any:
    """Return a pooled connection. Calling ``close()`` returns it to the pool."""
    try:
        return _get_pool().get_connection()
    except Error as exc:
        raise DatabaseOperationError("Unable to obtain a MySQL connection.") from exc


def initialize_database() -> bool:
    """Test database availability without allowing a database outage to stop Flask."""
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        return True
    except (DatabaseOperationError, Error) as exc:
        logger.warning("MySQL is unavailable; predictions will not be persisted: %s", exc)
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


@contextmanager
def transaction() -> Iterator[Any]:
    """Provide an atomic transaction with rollback and reliable connection return."""
    connection = None
    try:
        connection = get_connection()
        yield connection
        connection.commit()
    except DatabaseOperationError:
        if connection is not None:
            connection.rollback()
        raise
    except Error as exc:
        if connection is not None:
            connection.rollback()
        raise DatabaseOperationError("The MySQL transaction could not be completed.") from exc
    except Exception:
        if connection is not None:
            connection.rollback()
        raise
    finally:
        if connection is not None and connection.is_connected():
            connection.close()


def _execute(
    connection: Any, query: str, params: tuple = (), *, dictionary: bool = False
) -> tuple[Optional[Dict[str, Any]], int]:
    """Execute a parameterized statement and return its first row and last ID."""
    cursor = None
    try:
        cursor = connection.cursor(dictionary=dictionary)
        cursor.execute(query, params)
        row = cursor.fetchone() if dictionary else None
        return row, cursor.lastrowid
    except Error as exc:
        raise DatabaseOperationError("A MySQL query failed.") from exc
    finally:
        if cursor is not None:
            cursor.close()


def insert_user(email: str, full_name: str = "Anonymous Predictor", connection: Any = None) -> int:
    """Insert or reuse a user and return its ``user_id``.

    ``connection`` is optional for standalone use. The Flask request workflow
    passes a transaction connection so all writes succeed or fail together.
    """
    if connection is None:
        with transaction() as standalone_connection:
            return insert_user(email, full_name, connection=standalone_connection)

    _execute(
        connection,
        """
        INSERT INTO users (email, full_name)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE full_name = COALESCE(VALUES(full_name), full_name)
        """,
        (email, full_name),
    )
    row, _ = _execute(
        connection,
        "SELECT user_id FROM users WHERE email = %s",
        (email,),
        dictionary=True,
    )
    if row is None:
        raise DatabaseOperationError("The user record could not be found after insert.")
    return int(row["user_id"])


def insert_country(country_name: str, connection: Any = None) -> int:
    """Insert or reuse a country record and return its ``country_id``."""
    if connection is None:
        with transaction() as standalone_connection:
            return insert_country(country_name, connection=standalone_connection)

    _, country_id = _execute(
        connection,
        """
        INSERT INTO country (country_name)
        VALUES (%s)
        ON DUPLICATE KEY UPDATE country_id = LAST_INSERT_ID(country_id)
        """,
        (country_name,),
    )
    if not country_id:
        row, _ = _execute(
            connection,
            "SELECT country_id FROM country WHERE country_name = %s",
            (country_name,),
            dictionary=True,
        )
        if row is None:
            raise DatabaseOperationError("The country record could not be found after insert.")
        country_id = int(row["country_id"])
    return int(country_id)


def insert_input(
    user_id: int,
    country_id: int,
    life_expectancy: float,
    mean_years_of_schooling: float,
    expected_years_of_schooling: float,
    gni_per_capita: float,
    connection: Any = None,
) -> int:
    """Insert a set of submitted HDI indicators and return its ``input_id``."""
    if connection is None:
        with transaction() as standalone_connection:
            return insert_input(
                user_id,
                country_id,
                life_expectancy,
                mean_years_of_schooling,
                expected_years_of_schooling,
                gni_per_capita,
                connection=standalone_connection,
            )

    _, input_id = _execute(
        connection,
        """
        INSERT INTO hdi_input_data
            (user_id, country_id, life_expectancy, mean_years_of_schooling,
             expected_years_of_schooling, gni_per_capita)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            user_id,
            country_id,
            life_expectancy,
            mean_years_of_schooling,
            expected_years_of_schooling,
            gni_per_capita,
        ),
    )
    if not input_id:
        raise DatabaseOperationError("The HDI input record could not be created.")
    return int(input_id)


def save_prediction(
    input_id: int, model_id: int, predicted_hdi: float, connection: Any = None
) -> int:
    """Save one predicted HDI value and return its ``prediction_id``."""
    if connection is None:
        with transaction() as standalone_connection:
            return save_prediction(
                input_id, model_id, predicted_hdi, connection=standalone_connection
            )

    _, prediction_id = _execute(
        connection,
        """
        INSERT INTO hdi_prediction (input_id, model_id, predicted_hdi)
        VALUES (%s, %s, %s)
        """,
        (input_id, model_id, predicted_hdi),
    )
    if not prediction_id:
        raise DatabaseOperationError("The HDI prediction could not be saved.")
    return int(prediction_id)


def get_prediction_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Return a user's newest saved predictions with submitted input values."""
    safe_limit = max(1, min(int(limit), 100))
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT p.prediction_id, c.country_name, i.life_expectancy,
                   i.mean_years_of_schooling, i.expected_years_of_schooling,
                   i.gni_per_capita, p.predicted_hdi, p.predicted_at
            FROM hdi_prediction AS p
            INNER JOIN hdi_input_data AS i ON i.input_id = p.input_id
            INNER JOIN country AS c ON c.country_id = i.country_id
            WHERE i.user_id = %s
            ORDER BY p.predicted_at DESC
            LIMIT %s
            """,
            (user_id, safe_limit),
        )
        rows = cursor.fetchall()
        return rows
    except Error as exc:
        raise DatabaseOperationError("Prediction history could not be retrieved.") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def _get_or_create_model(connection: Any) -> int:
    """Register the bundled training dataset and linear regression model once."""
    _, dataset_id = _execute(
        connection,
        """
        INSERT INTO dataset (dataset_name, source_path, description)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE dataset_id = LAST_INSERT_ID(dataset_id)
        """,
        (
            "Human Development Index sample dataset",
            "Dataset/HumanDevelopmentIndex.csv",
            "Country-level health, education, income, and HDI indicators.",
        ),
    )
    _, model_id = _execute(
        connection,
        """
        INSERT INTO ml_model (dataset_id, model_name, algorithm, model_path, is_active)
        VALUES (%s, %s, %s, %s, TRUE)
        ON DUPLICATE KEY UPDATE model_id = LAST_INSERT_ID(model_id), is_active = TRUE
        """,
        (dataset_id, "HDI Linear Regression v1", "LinearRegression", "hdi_model.pkl"),
    )
    if not model_id:
        raise DatabaseOperationError("The ML model record could not be created.")
    return int(model_id)


def _record_session(connection: Any, user_id: int, session_token: str) -> int:
    """Create or update a browser session for the anonymous user workflow."""
    _, session_id = _execute(
        connection,
        """
        INSERT INTO user_session (user_id, session_token, last_seen_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP)
        ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), last_seen_at = CURRENT_TIMESTAMP,
                                session_id = LAST_INSERT_ID(session_id)
        """,
        (user_id, session_token),
    )
    if not session_id:
        raise DatabaseOperationError("The user session could not be recorded.")
    return int(session_id)


def store_prediction_record(
    *,
    country_name: str,
    life_expectancy: float,
    mean_years_of_schooling: float,
    expected_years_of_schooling: float,
    gni_per_capita: float,
    predicted_hdi: float,
    session_token: str,
) -> int:
    """Atomically store all normalized records for one Flask prediction request."""
    with transaction() as connection:
        user_id = insert_user("anonymous@hdi-predictor.local", connection=connection)
        _record_session(connection, user_id, session_token)
        country_id = insert_country(country_name, connection=connection)
        input_id = insert_input(
            user_id,
            country_id,
            life_expectancy,
            mean_years_of_schooling,
            expected_years_of_schooling,
            gni_per_capita,
            connection=connection,
        )
        model_id = _get_or_create_model(connection)
        return save_prediction(input_id, model_id, predicted_hdi, connection=connection)
