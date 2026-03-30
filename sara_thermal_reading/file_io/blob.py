import json
import logging
from io import BytesIO
from typing import List

import numpy as np
from azure.storage.blob import BlobServiceClient, ContainerClient, ContentSettings
from numpy.typing import NDArray
from PIL import Image, ImageFile

from sara_thermal_reading.file_io.tiff_loader import load_thermal_tiff_from_bytes

logger = logging.getLogger(__name__)


class BlobStore:
    def __init__(self, installation_code: str, connection_string: str) -> None:
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self.container_client: ContainerClient = (
            blob_service_client.get_container_client(installation_code)
        )

    def list_blobs_by_prefix(self, prefix: str) -> List[str]:
        """
        This function can take some time
        """
        blob_list = self.container_client.list_blobs(name_starts_with=prefix)
        blob_names: List[str] = [blob.name for blob in blob_list]
        return blob_names

    def check_if_exists(self, blob_name: str) -> bool:
        blob_client = self.container_client.get_blob_client(
            blob=blob_name,
        )
        is_existing: bool = blob_client.exists()
        return is_existing

    def download_bytes(self, blob_name: str) -> bytes:
        blob_client = self.container_client.get_blob_client(
            blob=blob_name,
        )
        blob: bytes = blob_client.download_blob().readall()
        return blob

    def download_image_array(self, blob_name: str) -> NDArray[np.uint8]:
        blob: bytes = self.download_bytes(blob_name)
        image: ImageFile.ImageFile = Image.open(BytesIO(blob))
        image_array: np.ndarray = np.array(image)
        return image_array

    def download_polygon(self, blob_name: str) -> list[tuple[int, int]]:
        blob: bytes = self.download_bytes(blob_name)
        blob_json: str = blob.decode("utf-8")
        polygon: list[tuple[int, int]] = json.loads(blob_json)
        return polygon

    def download_thermal_tiff(self, blob_name: str) -> np.ndarray:
        blob: bytes = self.download_bytes(blob_name)
        image: np.ndarray = load_thermal_tiff_from_bytes(blob)
        return image

    def upload_bytes(
        self,
        data: bytes,
        blob_name: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        blob_client = self.container_client.get_blob_client(
            blob=blob_name,
        )

        buffer_size = len(data)
        logger.debug(f"Uploading {buffer_size} bytes to blob {blob_name}")
        settings: ContentSettings = ContentSettings(content_type=content_type)

        try:
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=settings,
            )
            logger.debug(f"Successfully uploaded blob {blob_name}")
        except Exception as e:
            logger.error(f"Failed to upload blob {blob_name}: {e}")
            raise

    def upload_jpg(self, image: np.ndarray, blob_name: str) -> None:
        data_jpg: bytes = _np_image_to_jpg_bytes(image)
        logger.debug(f"Uploading image to {blob_name}")
        self.upload_bytes(data_jpg, blob_name, content_type="image/jpeg")

    def upload_polygon(self, polygon: list[tuple[int, int]], blob_name: str) -> None:
        polygon_json = json.dumps(polygon, indent=4)
        buf = BytesIO()
        buf.write(polygon_json.encode("utf-8"))
        buf.seek(0)
        data: bytes = buf.getvalue()
        self.upload_bytes(data, blob_name, content_type="application/json")


def _np_image_to_jpg_bytes(array: np.ndarray) -> bytes:
    img = Image.fromarray(array)

    if img.mode == "RGBA":
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    data: bytes = buf.getvalue()
    return data
