"""
Routes: /login, /logout and the before_app_request auth gate.

Auth is now opt-in and DB-user based (see routes/auth.py + core/auth.py):
- Implicit-owner mode: no accounts + no env gate → no login required.
- Login required once >1 account exists (or the CLU_USERNAME/CLU_PASSWORD env
  gate is set), validated against the users table.
"""
import pytest

from core.database import (
    create_api_token,
    create_user,
    get_user_by_username,
    seed_owner_if_needed,
)

LINK = {"link": "https://example.com/x.cbz"}

# Every /download endpoint that changes or reveals queue state.
GATED_DOWNLOAD_CALLS = [
    ("post", "/download"),
    ("get", "/download_status_all"),
    ("get", "/download_status/abc"),
    ("get", "/download_summary"),
    ("post", "/cancel_download/abc"),
    ("post", "/retry_download/abc"),
    ("post", "/dismiss_download/abc"),
    ("post", "/clear_downloads"),
    ("post", "/clear_failed_downloads"),
]


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _token_for(username):
    return create_api_token(get_user_by_username(username)["id"], name="extension")


def _assert_json_401(r):
    assert r.status_code == 401
    assert r.is_json
    assert "Location" not in r.headers
    assert r.headers.get("WWW-Authenticate") == "Bearer"


class TestImplicitOwnerMode:
    """With no real accounts and no env gate, the app requires no login."""

    def test_pages_accessible_without_login(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_login_page_redirects_to_index(self, client):
        r = client.get("/login")
        assert r.status_code == 302
        assert r.headers["Location"].endswith("/")

    def test_extension_needs_no_token(self, client):
        assert client.post("/download", json=LINK).status_code == 200
        assert client.post("/clear_downloads").status_code == 200

    def test_an_invalid_token_is_rejected(self, client):
        _assert_json_401(client.post("/download", json=LINK, headers=_bearer("nope")))

    def test_basic_auth_is_not_a_token(self, client):
        # Reverse-proxy credentials pass through; they are not ours to judge.
        r = client.post("/download", json=LINK, headers={"Authorization": "Basic abc"})
        assert r.status_code == 200


class TestLoginRequiredMultiUser:
    """More than one account exists → browser login is enforced."""

    @pytest.fixture(autouse=True)
    def _two_users(self, db_connection, monkeypatch):
        monkeypatch.delenv("CLU_USERNAME", raising=False)
        monkeypatch.delenv("CLU_PASSWORD", raising=False)
        create_user("owner", password="ownerpass", role="owner")
        create_user("reader", password="readerpass", role="reader")
        yield

    def test_unauthenticated_redirects_to_login(self, client):
        r = client.get("/")
        assert r.status_code == 302
        assert "/login" in r.headers["Location"]

    def test_login_page_renders(self, client):
        r = client.get("/login")
        assert r.status_code == 200
        assert "Sign In" in r.get_data(as_text=True)

    def test_correct_credentials_authenticate(self, client):
        r = client.post(
            "/login",
            data={"username": "reader", "password": "readerpass"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert client.get("/").status_code == 200

    def test_wrong_credentials_show_error(self, client):
        r = client.post("/login", data={"username": "reader", "password": "wrong"})
        assert r.status_code == 200
        assert "Invalid" in r.get_data(as_text=True)

    def test_next_param_redirect(self, client):
        r = client.post(
            "/login?next=/config",
            data={"username": "owner", "password": "ownerpass"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert r.headers["Location"].endswith("/config")

    def test_logout_clears_session(self, client):
        client.post("/login", data={"username": "owner", "password": "ownerpass"})
        assert client.get("/").status_code == 200

        r = client.get("/logout")
        assert r.status_code == 302
        assert "/login" in r.headers["Location"]

        # Session cleared → gated again.
        r = client.get("/")
        assert r.status_code == 302
        assert "/login" in r.headers["Location"]

    def test_opds_exempt_from_auth(self, client):
        r = client.get("/opds")
        assert "/login" not in r.headers.get("Location", "")

    def test_static_exempt_from_auth(self, client):
        r = client.get("/static/images/clu.png")
        assert r.status_code != 302 or "/login" not in r.headers.get("Location", "")

    # The /download family: the browser extension sends no session cookie,
    # so it needs an API token here -- and a JSON 401, never a redirect its
    # fetch would follow to a 200 login page.
    def test_anonymous_download_is_a_json_401(self, client):
        r = client.post("/download", json=LINK, follow_redirects=False)
        _assert_json_401(r)
        assert "Bearer" in r.get_json()["hint"]

    @pytest.mark.parametrize("method,path", GATED_DOWNLOAD_CALLS)
    def test_every_download_endpoint_needs_credentials(self, client, method, path):
        _assert_json_401(getattr(client, method)(path, follow_redirects=False))

    def test_preflight_and_help_page_stay_open(self, client):
        assert client.options("/download").status_code != 401
        assert client.get("/download").status_code == 200

    def test_invalid_token_is_401(self, client):
        _assert_json_401(client.post("/download", json=LINK, headers=_bearer("nope")))

    def test_owner_token_queues(self, client):
        r = client.post("/download", json=LINK, headers=_bearer(_token_for("owner")))
        assert r.status_code == 200

    def test_clerk_token_queues(self, client):
        create_user("clerk", password="clerkpass", role="clerk")
        r = client.post("/download", json=LINK, headers=_bearer(_token_for("clerk")))
        assert r.status_code == 200

    def test_reader_token_can_read_but_not_queue(self, client):
        token = _token_for("reader")
        r = client.post("/download", json=LINK, headers=_bearer(token))
        assert r.status_code == 403 and r.is_json
        assert client.get("/download_status_all", headers=_bearer(token)).status_code == 200

    def test_session_works_without_a_token(self, client):
        # The Status page calls these with its session cookie only.
        client.post("/login", data={"username": "owner", "password": "ownerpass"})
        assert client.post("/cancel_download/x").status_code == 200
        assert client.get("/download_status_all").status_code == 200

    def test_reader_session_cannot_manage_downloads(self, client):
        client.post("/login", data={"username": "reader", "password": "readerpass"})
        r = client.post("/clear_downloads")
        assert r.status_code == 403 and r.is_json


class TestEnvGateBackwardCompat:
    """The legacy CLU_USERNAME/CLU_PASSWORD env gate still forces login, now
    routed through a seeded owner account."""

    @pytest.fixture(autouse=True)
    def _env_gate(self, db_connection, monkeypatch):
        monkeypatch.setenv("CLU_USERNAME", "admin")
        monkeypatch.setenv("CLU_PASSWORD", "secret")
        seed_owner_if_needed()  # migrate env creds into a hashed owner account
        yield

    def test_unauthenticated_redirects_to_login(self, client):
        r = client.get("/")
        assert r.status_code == 302
        assert "/login" in r.headers["Location"]

    def test_anonymous_download_is_a_json_401(self, client):
        _assert_json_401(client.post("/download", json=LINK))

    def test_env_credentials_authenticate(self, client):
        r = client.post(
            "/login",
            data={"username": "admin", "password": "secret"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert client.get("/").status_code == 200
