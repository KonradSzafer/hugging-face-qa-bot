import logging
import os
import io
import json
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPIError

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("log_entry", "STRING", mode="REQUIRED"),
    ],
    write_disposition="WRITE_APPEND",
)


class BigQueryLoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        try:
            project_id = os.getenv("BIGQUERY_PROJECT_ID")
            dataset_id = os.getenv("BIGQUERY_DATASET_ID")
            table_id = os.getenv("BIGQUERY_TABLE_ID")
            print(f"project_id: {project_id}")
            print(f"dataset_id: {dataset_id}")
            print(f"table_id: {table_id}")
            service_account_info = json.loads(
                os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
                .replace('"', "")
                .replace("'", '"')
            )
            print(f"service_account_info: {service_account_info}")
            print(f"service_account_info type: {type(service_account_info)}")
            print(f"service_account_info keys: {service_account_info.keys()}")
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info
            )
            self.client = bigquery.Client(credentials=credentials, project=project_id)
            self.table_ref = self.client.dataset(dataset_id).table(table_id)
        except Exception as e:
            print(f"Error: {e}")
            self.handleError(e)

    def emit(self, record):
        try:
            recordstr = f"{self.format(record)}"
            body = io.BytesIO(recordstr.encode("utf-8"))
            job = self.client.load_table_from_file(
                body, self.table_ref, job_config=job_config
            )
            job.result()
        except GoogleAPIError as e:
            self.handleError(e)
        except Exception as e:
            self.handleError(e)

    def handleError(self, record):
        """
        Handle errors associated with logging.
        This method prevents logging-related exceptions from propagating.
        Optionally, implement more sophisticated error handling here.
        """
        if isinstance(record, logging.LogRecord):
            super().handleError(record)
        else:
            print(f"Logging error: {record}")


logger = logging.getLogger(__name__)


def setup_logger() -> None:
    """
    Logger setup.
    """
    logger.setLevel(logging.DEBUG)

    stream_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    bq_handler = BigQueryLoggingHandler()
    bq_handler.setFormatter(stream_formatter)
    logger.addHandler(bq_handler)
