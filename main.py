import json
import logging

import typer
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from sara_thermal_reading.config.logger import setup_logger
from sara_thermal_reading.config.open_telemetry import setup_open_telemetry
from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.main_thermal_workflow import (
    BlobStorageLocation,
    run_thermal_reading_workflow,
)

setup_logger()
logger = logging.getLogger(__name__)
setup_open_telemetry()
tracer = trace.get_tracer(settings.OTEL_SERVICE_NAME)


app = typer.Typer()


@app.command()
def run_thermal_reading(
    anonymized_blob_storage_location: str = typer.Option(
        ..., help="JSON string for anonymized data blob storage location"
    ),
    visualized_blob_storage_location: str = typer.Option(
        ..., help="JSON string for visualized data blob storage location"
    ),
    tag_id: str = typer.Option(..., help="Tag ID"),
    inspection_description: str = typer.Option(..., help="Inspection description"),
    installation_code: str = typer.Option(..., help="Installation code"),
    temperature_output_file: str = typer.Option(
        "/tmp/temperature_output.txt", help="Temperature output file path"
    ),
) -> None:
    try:
        anonymized_location = BlobStorageLocation.model_validate(
            json.loads(anonymized_blob_storage_location)
        )
        visualized_location = BlobStorageLocation.model_validate(
            json.loads(visualized_blob_storage_location)
        )
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON provided: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error parsing input: {e}")
        raise typer.Exit(code=1)

    with tracer.start_as_current_span(
        "cli.run",
        attributes={
            "src.container": anonymized_location.blob_container,
            "src.blob": anonymized_location.blob_name,
            "dst.container": visualized_location.blob_container,
            "dst.blob": visualized_location.blob_name,
        },
    ) as span:
        try:
            run_thermal_reading_workflow(
                anonymized_location,
                visualized_location,
                tag_id,
                inspection_description,
                installation_code,
                temperature_output_file,
            )
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise


if __name__ == "__main__":
    app()
