"""The browser interface is one static page served at / ; the API and its docs are unchanged."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home_serves_the_web_interface():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    page = resp.text
    for endpoint in ("/feed?", "/likes", "/follows", "/search?", "/users", "/posts"):
        assert endpoint in page, f"the page should call {endpoint}"


def test_web_interface_is_not_an_api_endpoint():
    paths = client.get("/openapi.json").json()["paths"]
    assert "/" not in paths
    assert {"/feed", "/search", "/likes"} <= set(paths)


def test_feed_limit_goes_up_to_350():
    limit = client.get("/openapi.json").json()["paths"]["/feed"]["get"]["parameters"]
    schema = next(p["schema"] for p in limit if p["name"] == "limit")
    assert schema["maximum"] == 350
    assert client.get("/feed", params={"user_id": 1, "limit": 351}).status_code == 422
