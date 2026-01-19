# gdrive_store.py
import io
from typing import Dict, List, Optional

import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]


def _drive_client():
    """
    Auth using TOML fields in Streamlit secrets under [gdrive_service_account].
    This avoids JSON parsing issues from multiline private keys.
    """
    sa = st.secrets.get("gdrive_service_account")
    if not sa:
        raise RuntimeError("Missing secrets table: [gdrive_service_account]")

    info = {
        "type": sa.get("type", "service_account"),
        "project_id": sa["project_id"],
        "private_key_id": sa["private_key_id"],
        # Google auth expects the key to contain real newlines; TOML triple quotes preserves them.
        "private_key": sa["private_key"],
        "client_email": sa["client_email"],
        "client_id": sa["client_id"],
        "auth_uri": sa.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": sa.get("token_uri", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": sa.get(
            "auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs"
        ),
        "client_x509_cert_url": sa["client_x509_cert_url"],
    }

    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_files_in_folder(folder_id: str) -> List[Dict]:
    service = _drive_client()
    q = f"'{folder_id}' in parents and trashed = false"
    fields = "files(id,name,modifiedTime,mimeType,size)"
    resp = service.files().list(q=q, fields=fields, orderBy="modifiedTime desc").execute()
    return resp.get("files", [])


def download_file_bytes(file_id: str) -> bytes:
    service = _drive_client()
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()


def upload_bytes_to_folder(
    folder_id: str,
    filename: str,
    content: bytes,
    mime_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    overwrite: bool = True,
) -> Dict:
    """
    Upload bytes to Drive folder. If overwrite=True and same filename exists, replace it.
    """
    service = _drive_client()

    existing_id: Optional[str] = None
    if overwrite:
        safe_name = filename.replace("'", "")
        q = f"'{folder_id}' in parents and trashed = false and name = '{safe_name}'"
        resp = service.files().list(q=q, fields="files(id,name)").execute()
        files = resp.get("files", [])
        if files:
            existing_id = files[0]["id"]

    media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=True)

    if existing_id:
        return service.files().update(fileId=existing_id, media_body=media).execute()

    file_metadata = {"name": filename, "parents": [folder_id]}
    return service.files().create(body=file_metadata, media_body=media, fields="id,name").execute()
