import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.skipif(not os.path.exists('tests/sample.pdf'), reason='Sample PDF not found')
def test_ingest_single_pdf():
    with open('tests/sample.pdf', 'rb') as f:
        response = client.post(
            '/api/ingest',
            files={'files': ('sample.pdf', f, 'application/pdf')}
        )
    assert response.status_code == 200
    data = response.json()
    assert 'results' in data
    assert data['results'][0]['filename'] == 'sample.pdf'
    assert data['results'][0]['status'] == 'success' or data['results'][0]['status'] == 'error' 