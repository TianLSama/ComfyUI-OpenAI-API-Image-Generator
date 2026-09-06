"""响应解析纯单元测试（T3 · RED 阶段）。

被测量模块 cfoapi.parsing / cfoapi.errors 尚未实现，本文件在导入期即应
失败（ModuleNotFoundError/ImportError），符合 TDD 红灯预期。
全部为纯函数断言，不触碰任何 HTTP，也不使用 mock_server 夹具。
"""
from __future__ import annotations

import pytest

from cfoapi.errors import MalformedResponseError, OpenAPIConfigError
from cfoapi.parsing import (
    ImageItem,
    apply_system_prompt,
    build_request_body,
    collect_set_widgets,
    extract_image_items,
    normalize_size,
    parse_error_envelope,
    parse_model_list,
)


def _basic(**overrides: object) -> dict[str, object]:
    """构造与 INPUT_TYPES 默认值一致的基本参数映射（控件全空 / count=1）。"""
    base: dict[str, object] = {
        "size": "",
        "quality": "",
        "output_format": "",
        "background": "",
        "moderation": "",
        "negative_prompt": "",
        "count": 1,
    }
    base.update(overrides)
    return base


# ---------- parse_model_list ----------


def test_parse_model_list_preserves_order() -> None:
    # Given: data[] 含三个有序 id
    payload = {"data": [{"id": "c"}, {"id": "a"}, {"id": "b"}]}
    # Then: 返回顺序与 data 一致
    assert parse_model_list(payload) == ["c", "a", "b"]


def test_parse_model_list_skips_missing_and_non_str_id() -> None:
    # Given: 混入缺 id 与 id 非字符串（123）的条目
    payload = {"data": [{"id": "ok1"}, {"nope": 1}, {"id": 123}, {"id": "ok2"}]}
    # Then: 仅保留合法字符串 id，顺序不变
    assert parse_model_list(payload) == ["ok1", "ok2"]


def test_parse_model_list_raises_when_data_missing() -> None:
    # Then: dict 但缺 "data" 键 → 畸形响应
    with pytest.raises(MalformedResponseError):
        parse_model_list({"object": "list"})


def test_parse_model_list_raises_when_payload_not_dict() -> None:
    # Then: 载荷非 dict（列表）→ 畸形响应
    with pytest.raises(MalformedResponseError):
        parse_model_list(["not", "a", "dict"])


# ---------- extract_image_items ----------


def test_extract_image_items_b64_single() -> None:
    # Then: data[].b64_json → kind=b64
    assert extract_image_items({"data": [{"b64_json": "AAA"}]}) == [
        ImageItem(kind="b64", value="AAA")
    ]


def test_extract_image_items_url_single() -> None:
    # Then: data[].url → kind=url
    assert extract_image_items({"data": [{"url": "http://x/y.png"}]}) == [
        ImageItem(kind="url", value="http://x/y.png")
    ]


def test_extract_image_items_b64_wins_over_url_within_entry() -> None:
    # Given: 同一 data 条目同时带 b64_json 与 url
    items = extract_image_items({"data": [{"b64_json": "AAA", "url": "http://x/y.png"}]})
    # Then: 仅产出一个条目且 b64 优先
    assert len(items) == 1
    assert items[0].kind == "b64"


def test_extract_image_items_siliconflow_images_shape() -> None:
    # Then: 顶层 images[].url（SiliconFlow 形状）被识别为 url
    assert extract_image_items({"images": [{"url": "http://x/s.png"}]}) == [
        ImageItem(kind="url", value="http://x/s.png")
    ]


def test_extract_image_items_raises_when_data_empty() -> None:
    # Then: data 为空列表 → 畸形响应
    with pytest.raises(MalformedResponseError):
        extract_image_items({"data": []})


def test_extract_image_items_raises_when_no_image_found() -> None:
    # Then: 既无 data 也无 images → 找不到任何图像
    with pytest.raises(MalformedResponseError):
        extract_image_items({"foo": 1})


# ---------- parse_error_envelope ----------


def test_parse_error_envelope_openai_error_message() -> None:
    # When: 标准 OpenAI {"error":{"message":m}} 包络
    msg = parse_error_envelope(401, {"error": {"message": "Invalid API key provided."}})
    # Then: 提取出 message 文本
    assert "Invalid API key" in msg


def test_parse_error_envelope_gateway_code_and_message() -> None:
    # When: 网关风格 {"code":c,"message":m}
    msg = parse_error_envelope(
        400, {"code": "insufficient_credit", "message": "Account has no credit"}
    )
    # Then: code 与 message 均出现在结果中
    assert "insufficient_credit" in msg
    assert "Account has no credit" in msg


def test_parse_error_envelope_non_dict_falls_back_with_status() -> None:
    # When: 非 dict（HTML 字符串）
    msg = parse_error_envelope(502, "<html>Bad Gateway</html>")
    # Then: 兜底文案包含状态码
    assert "502" in msg


def test_parse_error_envelope_dict_without_message_uses_status() -> None:
    # When: dict 但无可提取 message
    msg = parse_error_envelope(500, {"weird": True})
    # Then: 通用文案包含状态码
    assert "500" in msg


# ---------- build_request_body ----------


def test_build_request_body_empty_params_yields_defaults() -> None:
    # Then: params_json 为空串 → 仅基础三键，response_format 取默认 b64_json
    assert build_request_body("p", "m", "") == {
        "model": "m",
        "prompt": "p",
        "response_format": "b64_json",
    }


def test_build_request_body_merges_extra_params() -> None:
    # When: 合入 size/quality
    body = build_request_body("p", "m", '{"size":"64x64","quality":"low"}')
    # Then: 新键并入，基础键保留
    assert body["size"] == "64x64"
    assert body["quality"] == "low"
    assert body["model"] == "m"
    assert body["prompt"] == "p"
    assert body["response_format"] == "b64_json"


def test_build_request_body_locks_model_and_prompt() -> None:
    # When: params 试图覆盖 model/prompt
    body = build_request_body("p", "m", '{"prompt":"evil","model":"evil"}')
    # Then: 二者被锁定，不可覆盖
    assert body["prompt"] == "p"
    assert body["model"] == "m"


def test_build_request_body_allows_response_format_override() -> None:
    # When: params 覆盖 response_format（允许）
    body = build_request_body("p", "m", '{"response_format":"url"}')
    # Then: 覆盖生效
    assert body["response_format"] == "url"


def test_build_request_body_raises_on_invalid_json_with_params_hint() -> None:
    # When/Then: 非法 JSON 抛错，信息含 "params"
    with pytest.raises(OpenAPIConfigError) as exc_info:
        build_request_body("p", "m", "{bad json")
    assert "params" in str(exc_info.value)


def test_build_request_body_raises_on_json_list() -> None:
    # Then: JSON 数组（非对象）被拒绝
    with pytest.raises(OpenAPIConfigError):
        build_request_body("p", "m", "[1,2]")


def test_build_request_body_raises_on_json_null() -> None:
    # Then: JSON null（非对象）被拒绝
    with pytest.raises(OpenAPIConfigError):
        build_request_body("p", "m", "null")


# ---------- apply_system_prompt ----------


def test_apply_system_prompt_prepends_when_present() -> None:
    # Then: 非空 system 前置并换行拼接
    assert apply_system_prompt("user", "sys") == "sys\nuser"


def test_apply_system_prompt_returns_prompt_when_system_empty() -> None:
    # Then: 空 system 原样返回 prompt
    assert apply_system_prompt("user", "") == "user"


def test_apply_system_prompt_returns_prompt_when_system_whitespace() -> None:
    # Then: 纯空白 system 视作空，prompt 不变
    assert apply_system_prompt("user", "   ") == "user"


# ---------- normalize_size（v0.3 基本参数控件 · size 归一化） ----------


@pytest.mark.parametrize(
    "raw",
    ["1024x1024", "1024X1024", "1024*1024", "1024×1024", "1024 ✕ 1024", "1024 ✖ 1024", "  1024 x 1024  "],
)
def test_normalize_size_accepts_all_separator_and_whitespace_shapes(raw: str) -> None:
    # When/Then: 六种分隔符与两侧空白均被接受，输出按协议分隔符归一
    assert normalize_size(raw, "x") == "1024x1024"
    assert normalize_size(raw, "*") == "1024*1024"


def test_normalize_size_passes_auto_through_case_insensitively() -> None:
    # Then: 'auto'（含大小写与外围空白）原样返回小写 'auto'，由协议侧决定放行与否
    assert normalize_size("auto", "x") == "auto"
    assert normalize_size(" Auto ", "x") == "auto"


@pytest.mark.parametrize(
    "raw",
    ["1024", "10x", "x10", "1x2", "123456x1024", "1024+1024", "wide", "1024 x", "autox", "1024x1024x1024"],
)
def test_normalize_size_rejects_invalid_shapes_with_format_hint(raw: str) -> None:
    # When/Then: 非尺寸非 auto → 配置错误且消息含格式提示
    with pytest.raises(OpenAPIConfigError) as exc_info:
        normalize_size(raw, "x")
    assert "Invalid size" in str(exc_info.value)
    assert "1024x1024" in str(exc_info.value)


# ---------- collect_set_widgets（「已设置」判定） ----------


def test_collect_set_widgets_keeps_only_set_controls() -> None:
    # Given: 空白/空串控件与 count=1 均未设置，其余为已设置
    entries = collect_set_widgets(
        _basic(size=" 1024x1024 ", quality="", output_format="png", background="   ", negative_prompt="ugly")
    )
    # Then: 仅保留 trim 后非空的字符串控件；值已 trim
    assert entries == {"size": "1024x1024", "output_format": "png", "negative_prompt": "ugly"}


def test_collect_set_widgets_count_semantics_and_empty_input() -> None:
    # Then: count>1 才算已设置；count=1 / None / 空映射 → 无贡献
    assert collect_set_widgets(_basic(count=2)) == {"count": 2}
    assert collect_set_widgets(_basic()) == {}
    assert collect_set_widgets(None) == {}
    assert collect_set_widgets({}) == {}


# ---------- build_request_body × basic_params（openai 侧语义） ----------


def test_build_request_body_maps_widgets_to_openai_top_level_keys() -> None:
    # When: 除 negative_prompt 外全部控件设置（size 用 * 形式输入，验证归一为 x）
    body = build_request_body(
        "p",
        "m",
        "",
        basic_params=_basic(
            size="1664*928",
            quality="high",
            output_format="webp",
            background="transparent",
            moderation="low",
            count=3,
        ),
    )
    # Then: 控件→顶层键（count→n），size 归一为协议正典 WxH
    assert body["size"] == "1664x928"
    assert body["quality"] == "high"
    assert body["output_format"] == "webp"
    assert body["background"] == "transparent"
    assert body["moderation"] == "low"
    assert body["n"] == 3


def test_build_request_body_all_defaults_is_identical_to_v02() -> None:
    # Then: 全默认控件 + 空 params → 请求体与 v0.2 逐键一致（回归契约）
    expected = {"model": "m", "prompt": "p", "response_format": "b64_json"}
    assert build_request_body("p", "m", "", basic_params=_basic()) == expected
    assert build_request_body("p", "m", "", basic_params=None) == expected


def test_build_request_body_json_overrides_same_name_widget_keys() -> None:
    # Given: 控件与 JSON 同名（size），且 JSON 值是非规范形式
    # When/Then: JSON 优先且逐字节透传（不归一、不改写）
    body = build_request_body("p", "m", '{"size":"2048*1152"}', basic_params=_basic(size="1024x1024"))
    assert body["size"] == "2048*1152"


def test_build_request_body_widgets_fill_gaps_json_channel_keeps_working() -> None:
    # When: 控件提供 quality，JSON 提供无控件的高级键
    body = build_request_body("p", "m", '{"watermark":false}', basic_params=_basic(quality="low"))
    # Then: 两通道并存互不干扰
    assert body["quality"] == "low"
    assert body["watermark"] is False


def test_build_request_body_rejects_dashscope_only_negative_prompt() -> None:
    # When/Then: openai 协议下 negative_prompt 控件被设置 → 协议错配报错并点名控件
    with pytest.raises(OpenAPIConfigError) as exc_info:
        build_request_body("p", "m", "", basic_params=_basic(negative_prompt="ugly hands"))
    assert "negative_prompt" in str(exc_info.value)


def test_build_request_body_json_channel_allows_negative_prompt() -> None:
    # Then: JSON 通道不受限——openai 用户仍可经 params 传 negative_prompt
    body = build_request_body("p", "m", '{"negative_prompt":"blurry"}')
    assert body["negative_prompt"] == "blurry"


def test_build_request_body_invalid_widget_size_raises_even_without_params() -> None:
    # When/Then: 控件 size 非法 → 报错含格式提示（哪怕 params 为空）
    with pytest.raises(OpenAPIConfigError) as exc_info:
        build_request_body("p", "m", "", basic_params=_basic(size="1024 by 1024"))
    assert "Invalid size" in str(exc_info.value)
