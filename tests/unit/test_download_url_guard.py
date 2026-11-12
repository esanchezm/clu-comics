"""What the download pipeline may fetch, and which headers go with it.

POST /download is reachable without a login in implicit-owner mode, and the
resolver and downloaders follow redirects. Without these guards a caller could
make CLU fetch a LAN or cloud-metadata address, send the owner's custom
headers (GetComics access credentials) to any host, or name the file.

The helpers live in core.download_utils because api.py cannot be imported by
the suite; tests/unit/test_download_endpoint_hardening.py pins that api.py
calls them. No test here performs real DNS or opens a socket.
"""
import socket

import pytest
import requests
from requests.adapters import BaseAdapter
from requests.models import Response

from core import download_utils
from core.download_utils import (
    UnresolvableHostError,
    fetch_url,
    UnsafeURLError,
    assert_public_url,
    harden_session,
    is_policy_refusal,
    sensitive_custom_headers,
    parse_download_request,
    request_headers_for,
    url_host_is,
)


def _resolver(*addresses):
    def resolve(host, port, type=None):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]
    return resolve


def _no_dns(*args, **kwargs):
    raise AssertionError("resolver must not be called")


class TestAssertPublicUrl:

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "ftp://x/y",
        "gopher://x",
        "//x/y",
        "",
        None,
        123,
        "http://",
        "http://localhost/",
        "http://foo.localhost/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://10.0.0.1/",
        "http://192.168.1.10:8080/",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://[fd00::1]/",
        "http://[fe80::1]/",
        "http://100.64.0.1/",
        "http://0.0.0.0/",
        "http://[::ffff:127.0.0.1]/",
        "http://[2002:7f00:1::]/",   # 6to4 wrapping 127.0.0.1
        "http://224.0.0.1/",
        "http://example.com:99999/",
        # urllib.parse says example.com; urllib3, which requests connects
        # through, says 127.0.0.1.
        "http://127.0.0.1\\@example.com/",
        "http://127.0.0.1:80\\@pixeldrain.com/",
        "http://169.254.169.254\\@example.com/latest/meta-data/",
        "http://exa mple.com/",
        "http://example.com\t/",
        "http://[64:ff9b::a00:5]/",   # NAT64 to 10.0.0.5
        "http://[::127.0.0.1]/",      # IPv4-compatible
    ])
    def test_refuses(self, url):
        with pytest.raises(UnsafeURLError):
            assert_public_url(url, resolver=_no_dns)

    def test_accepts_a_public_literal(self):
        assert assert_public_url("https://8.8.8.8/x", resolver=_no_dns) == "https://8.8.8.8/x"

    def test_a_public_name_passes(self):
        url = "https://example.com/x.cbz"
        assert assert_public_url(url, resolver=_resolver("93.184.216.34")) == url

    def test_one_private_record_refuses_the_name(self):
        with pytest.raises(UnsafeURLError):
            assert_public_url("https://example.com/",
                              resolver=_resolver("93.184.216.34", "10.0.0.1"))

    def test_a_resolution_failure_refuses_but_is_retryable(self):
        def fail(*args, **kwargs):
            raise socket.gaierror("nope")
        with pytest.raises(UnresolvableHostError, match="could not resolve") as exc:
            assert_public_url("https://example.com/", resolver=fail)
        # A DNS hiccup must not cost the download its retries.
        assert not is_policy_refusal(exc.value)

    def test_a_private_address_is_a_policy_refusal(self):
        with pytest.raises(UnsafeURLError) as exc:
            assert_public_url("http://10.0.0.1/")
        assert is_policy_refusal(exc.value)
        assert not is_policy_refusal(RuntimeError("x"))

    @pytest.mark.parametrize("url", [
        "http://%31%32%37%2e0.0.1/",
        "http://127%2e0%2e0%2e1/",
    ])
    def test_percent_encoded_literals_are_refused_without_dns(self, url):
        with pytest.raises(UnsafeURLError) as exc:
            assert_public_url(url, resolve=False)
        assert is_policy_refusal(exc.value)

    def test_fetch_url_quotes_like_requests(self):
        assert fetch_url("https://cdn.example.com/a/", "Batman 001 (2020).cbz") == (
            "https://cdn.example.com/a/Batman%20001%20(2020).cbz")
        # Quoted, a backslash becomes userinfo, and both parsers -- and so the
        # connection -- agree the host is example.com.
        from urllib3.util import parse_url
        quoted = fetch_url("http://127.0.0.1\\@example.com/")
        assert parse_url(quoted).host == "example.com"
        assert assert_public_url(quoted, resolver=_resolver("93.184.216.34")) == quoted

    @pytest.mark.parametrize("url", [
        "https://user:pw@getcomics.org/x",
        "https://getcomics.org./x",
        "https://xn--bcher-kva.example/x",
        "https://8.8.8.8:8443/x?a=b#c",
    ])
    def test_ordinary_urls_are_not_ambiguous(self, url):
        assert assert_public_url(url, resolver=_resolver("93.184.216.34")) == url

    def test_resolve_false_skips_dns(self):
        assert assert_public_url("https://example.com/", resolve=False, resolver=_no_dns)

    def test_resolve_false_still_checks_literals_and_schemes(self):
        for url in ("http://127.0.0.1/", "file:///x"):
            with pytest.raises(UnsafeURLError):
                assert_public_url(url, resolve=False)

    def test_it_is_a_value_error(self):
        assert issubclass(UnsafeURLError, ValueError)


class TestUrlHostIs:

    @pytest.mark.parametrize("url,expected", [
        ("https://pixeldrain.com/u/a", True),
        ("https://cdn.pixeldrain.com/x", True),
        ("https://PixelDrain.COM/x", True),
        ("http://10.0.0.1/?pixeldrain.com", False),
        ("https://notpixeldrain.com/", False),
        ("https://pixeldrain.com.evil.example/", False),
        ("https://evil.example/pixeldrain.com", False),
        ("not a url", False),
        (None, False),
    ])
    def test_matches_the_host_only(self, url, expected):
        assert url_host_is(url, "pixeldrain.com") is expected

    def test_any_of_several(self):
        assert url_host_is("https://mega.co.nz/x", "mega.nz", "mega.co.nz")


class TestRequestHeaders:

    @pytest.fixture
    def setting(self, monkeypatch):
        value = {"HEADERS": '{"X-Pass": " secret ", "Cookie": "c=1"}'}
        monkeypatch.setattr("core.settings.get_setting", lambda key: value.get(key, ""))
        return value

    @pytest.mark.parametrize("url", ["https://getcomics.org/x", "https://sub.comicfiles.ru/x"])
    def test_sent_to_getcomics_hosts(self, setting, url):
        hdrs = request_headers_for(url, internal=False)
        assert hdrs["X-Pass"] == "secret"
        assert hdrs["Cookie"] == "c=1"
        assert "User-Agent" in hdrs

    @pytest.mark.parametrize("url", [
        "https://evil.example/?getcomics.org",
        "https://getcomics.org.evil.example/",
        "https://pixeldrain.com/u/x",
        "https://mega.nz/file/x",
    ])
    def test_not_sent_anywhere_else(self, setting, url):
        assert set(request_headers_for(url, internal=False)) == {"User-Agent"}

    def test_not_sent_for_internal_downloads(self, setting):
        assert set(request_headers_for("https://getcomics.org/x", internal=True)) == {"User-Agent"}

    @pytest.mark.parametrize("raw", ["{not json", '["X-Pass", "y"]', '{"X": 1}', "", "   "])
    def test_bad_settings_yield_the_user_agent_only(self, setting, raw):
        setting["HEADERS"] = raw
        assert set(request_headers_for("https://getcomics.org/x", internal=False)) == {"User-Agent"}

    def test_read_per_call(self, setting):
        url = "https://getcomics.org/x"
        assert "X-Pass" in request_headers_for(url, internal=False)
        setting["HEADERS"] = '{"X-Other": "v"}'
        hdrs = request_headers_for(url, internal=False)
        assert "X-Other" in hdrs and "X-Pass" not in hdrs


class _RedirectAdapter(BaseAdapter):
    """Canned redirect chain, no sockets: url -> (status, location)."""

    def __init__(self, routes):
        super().__init__()
        self.routes = routes
        self.seen = []

    def send(self, request, **kwargs):
        self.seen.append((request.url, dict(request.headers)))
        status, location = self.routes.get(request.url, (200, None))
        resp = Response()
        resp.status_code = status
        resp.url = request.url
        resp.request = request
        resp._content = b"ok"
        if location:
            resp.headers["Location"] = location
        return resp

    def close(self):
        pass


def _scraper():
    cloudscraper = pytest.importorskip("cloudscraper")
    return cloudscraper.create_scraper()


@pytest.fixture(params=["requests", "cloudscraper"])
def make_session(request, monkeypatch):
    # Public hosts in these chains are resolved by the fake resolver below.
    monkeypatch.setattr(
        download_utils.socket, "getaddrinfo", _resolver("93.184.216.34"))

    def build(routes):
        s = requests.Session() if request.param == "requests" else _scraper()
        adapter = _RedirectAdapter(routes)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers["X-Pass"] = "secret"
        harden_session(s, sensitive_headers={"X-Pass": "secret"})
        return s, adapter
    return build


class TestHardenSession:

    def test_custom_header_is_dropped_off_getcomics(self, make_session):
        s, adapter = make_session({
            "https://getcomics.org/dl": (302, "https://evil.example/file"),
        })
        s.get("https://getcomics.org/dl")
        (url1, h1), (url2, h2) = adapter.seen
        assert h1.get("X-Pass") == "secret"
        assert url2 == "https://evil.example/file"
        assert "X-Pass" not in h2
        assert "User-Agent" in h2

    def test_redirect_with_a_raw_space_is_followed(self, make_session):
        s, adapter = make_session({
            "https://getcomics.org/dl": (302, "https://cdn.example.com/f/Batman 001.cbz"),
        })
        s.get("https://getcomics.org/dl")
        assert adapter.seen[-1][0] == "https://cdn.example.com/f/Batman%20001.cbz"

    def test_same_host_redirect_keeps_it(self, make_session):
        s, adapter = make_session({
            "https://getcomics.org/a": (302, "/b"),
        })
        s.get("https://getcomics.org/a")
        assert adapter.seen[1][1].get("X-Pass") == "secret"

    def test_redirect_to_a_private_address_is_never_requested(self, make_session):
        s, adapter = make_session({
            "https://getcomics.org/dl": (302, "http://127.0.0.1/admin"),
        })
        with pytest.raises(UnsafeURLError):
            s.get("https://getcomics.org/dl")
        assert [u for u, _ in adapter.seen] == ["https://getcomics.org/dl"]

    def test_a_later_hop_is_checked_too(self, make_session):
        s, adapter = make_session({
            "https://getcomics.org/a": (302, "https://mirror.example/b"),
            "https://mirror.example/b": (301, "http://169.254.169.254/latest"),
        })
        with pytest.raises(UnsafeURLError):
            s.get("https://getcomics.org/a")
        assert "169.254" not in " ".join(u for u, _ in adapter.seen)


class TestSensitiveCustomHeaders:

    @pytest.fixture(autouse=True)
    def setting(self, monkeypatch):
        value = {"HEADERS": '{"X-Pass": "secret", "User-Agent": "mine", "Referer": "r"}'}
        monkeypatch.setattr("core.settings.get_setting", lambda key: value.get(key, ""))

    def test_only_headers_actually_sent(self):
        assert sensitive_custom_headers({"X-Pass": "secret"}) == {"X-Pass": "secret"}
        assert sensitive_custom_headers({"X-Pass": "other"}) == {}
        assert sensitive_custom_headers({}) == {}

    def test_never_the_session_user_agent_or_referer(self):
        sent = {"X-Pass": "secret", "User-Agent": "mine", "Referer": "r"}
        assert set(sensitive_custom_headers(sent)) == {"X-Pass"}


class TestParseDownloadRequest:

    @pytest.mark.parametrize("data", [
        None,
        ["https://example.com/x.cbz"],
        {},
        {"link": {"url": "https://example.com"}},
        {"link": ""},
        {"link": "https://example.com/" + "a" * 5000},
        {"link": "file:///etc/passwd"},
        {"link": "http://127.0.0.1:5577/x.cbz"},
        {"link": "https://example.com/x.cbz", "dest_filename": {"a": 1}},
        {"link": "https://example.com/x.cbz", "dest_filename": "x" * 300},
    ])
    def test_rejects(self, data):
        with pytest.raises(ValueError):
            parse_download_request(data)

    def test_dest_filename_is_reduced_to_a_name(self):
        assert parse_download_request(
            {"link": "https://example.com/x", "dest_filename": "../x.cbz"}
        ) == ("https://example.com/x", "x.cbz")

    def test_ordinary_request(self):
        assert parse_download_request({"link": " https://example.com/x "}) == (
            "https://example.com/x", None)
