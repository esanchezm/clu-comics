"""api.py calls the download guards in core.download_utils.

api.py cannot be imported by the suite (worker threads, cloudscraper, DB at
import), so the guards' behaviour is tested in test_download_url_guard.py and
test_download_name_claim.py, and this file pins that api.py really calls them:

* POST /download validates its body (scheme, private literals, the name);
* the resolver checks every hop before requesting it, and a refusal is not
  swallowed as "unresolvable";
* process_download checks the link and the resolved URL, routes by host
  rather than by substring, and marks a refusal as not worth auto-retrying;
* downloader sessions check their own redirects;
* the owner's custom headers are built per URL, never frozen at import.
"""
import ast
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
API_PATH = os.path.join(PROJECT_ROOT, "api.py")


@pytest.fixture(scope="module")
def tree():
    with open(API_PATH, encoding="utf-8") as fh:
        return ast.parse(fh.read())


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"{name} not found in api.py")


def _call_names(node):
    names = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                names.append(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                names.append(n.func.attr)
    return names


def _contains_call(node, name):
    return name in _call_names(node)


class TestDownloadRoute:

    def test_body_goes_through_the_parser(self, tree):
        assert _contains_call(_func(tree, "download"), "parse_download_request")

    def test_no_raw_link_subscript(self, tree):
        for n in ast.walk(_func(tree, "download")):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant):
                assert n.slice.value != "link"

    def test_a_non_json_body_is_a_400_not_a_500(self, tree):
        calls = [n for n in ast.walk(_func(tree, "download"))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "get_json"]
        assert calls and all(
            any(k.arg == "silent" and getattr(k.value, "value", None) is True
                for k in c.keywords) for c in calls)


class TestResolver:

    def test_every_hop_is_checked_outside_the_swallowing_try(self, tree):
        fn = _func(tree, "resolve_final_url")
        loop = next(n for n in fn.body if isinstance(n, ast.For))
        # A direct statement of the loop body, not nested in any try.
        assert any(
            isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and getattr(stmt.value.func, "id", None) == "assert_public_url"
            for stmt in loop.body
        )
        for t in (n for n in ast.walk(fn) if isinstance(n, ast.Try)):
            assert "assert_public_url" not in _call_names(ast.Module(body=t.body, type_ignores=[]))

    def test_getcomics_is_matched_by_host(self, tree):
        src = ast.unparse(_func(tree, "resolve_final_url"))
        assert "'getcomics.org' in" not in src and '"getcomics.org" in' not in src
        assert "url_host_is" in src

    def test_no_hdrs_parameter(self, tree):
        fn = _func(tree, "resolve_final_url")
        params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        assert "hdrs" not in params and "internal" in params


class TestProcessDownload:

    def test_link_and_resolved_url_are_checked(self, tree):
        assert _call_names(_func(tree, "process_download")).count("assert_public_url") >= 2

    def test_routing_is_by_host_not_substring(self, tree):
        fn = _func(tree, "process_download")
        for n in ast.walk(fn):
            if isinstance(n, ast.Compare) and any(isinstance(op, ast.In) for op in n.ops):
                if isinstance(n.left, ast.Constant) and isinstance(n.left.value, str):
                    right = ast.unparse(n.comparators[0])
                    assert "final_url" not in right and "try_url" not in right, ast.unparse(n)
        # The skip-resolve test was a generator of substring tests.
        src = ast.unparse(fn)
        assert "in try_url.lower()" not in src

    def test_a_refusal_disables_the_auto_retry(self, tree):
        fn = _func(tree, "process_download")
        for handler in (n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)):
            src = ast.unparse(handler)
            if "UnsafeURLError" in src and "'no_retry'" in src:
                # ...but not a DNS failure, which is transient.
                assert "is_policy_refusal" in src
                return
        pytest.fail("process_download must flag no_retry when it sees UnsafeURLError")

    def test_downloaders_get_headers_built_for_the_url(self, tree):
        fn = _func(tree, "process_download")
        assigned = {
            t.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and getattr(n.value.func, "id", None) == "request_headers_for"
            for t in n.targets if isinstance(t, ast.Name)
        }
        hdrs = [k.value for n in ast.walk(fn) if isinstance(n, ast.Call)
                for k in n.keywords if k.arg == "hdrs"]
        assert hdrs
        for value in hdrs:
            ok = (isinstance(value, ast.Name) and value.id in assigned) or (
                isinstance(value, ast.Call)
                and getattr(value.func, "id", None) == "request_headers_for")
            assert ok, ast.unparse(value)


class TestSessions:

    @pytest.mark.parametrize("name", ["_requests_session", "download_getcomics"])
    def test_sessions_check_their_redirects(self, tree, name):
        assert _contains_call(_func(tree, name), "harden_session")

    def test_getcomics_downloader_keeps_the_refusal_type(self, tree):
        src = ast.unparse(_func(tree, "download_getcomics"))
        assert "UnsafeURLError" in src and "is_policy_refusal" in src

    def test_flaresolverr_only_solves_getcomics_hosts(self, tree):
        """Its browser follows redirects and runs scripts outside every check,
        and any server can fake a challenge response."""
        fn = _func(tree, "download_getcomics")
        gates = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                 and "clear_challenge" in _call_names(ast.Module(body=n.body, type_ignores=[]))]
        assert any("url_host_is(url, *GETCOMICS_HOSTS)" in ast.unparse(g.test)
                   for g in gates)


class TestCustomHeadersAreNotFrozen:

    def test_no_import_time_read(self, tree):
        for stmt in tree.body:
            src = ast.unparse(stmt)
            assert "custom_headers" not in src or isinstance(
                stmt, (ast.FunctionDef, ast.ImportFrom)), src.splitlines()[0]
        names = {t.id for stmt in tree.body if isinstance(stmt, ast.Assign)
                 for t in stmt.targets if isinstance(t, ast.Name)}
        assert not names & {"custom_headers_str", "headers", "default_headers"}
