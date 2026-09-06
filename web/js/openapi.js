// ComfyUI-OpenAPI 前端扩展：在 OpenAPIImageGenerator 节点上提供“获取模型列表”按钮
// 注意：必须用绝对路径导入。本文件位于 web/js/（两层深），服务路径为
// /extensions/<pkg>/js/openapi.js —— 相对路径 "../../scripts/app.js" 会错误解析到
// /extensions/scripts/app.js 而 404（实机 QA 已验证）。绝对路径与嵌套深度无关。
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_CLASS = "OpenAPIImageGenerator";
const ROUTE_FETCH = "/comfyui_openapi/fetch_models";
const ROUTE_CACHE = "/comfyui_openapi/cache";

// v0.3：参数模板 JSON 骨架（“参数模板”按钮写入 params 文本域，2 空格缩进）。
// 键顺序即模板顺序；JSON 中同名键优先于交互控件（后端已实现该覆盖规则）。
const PARAMS_TEMPLATES = {
  openai: {
    size: "1024x1024",
    quality: "high",
    output_format: "png",
    background: "opaque",
    moderation: "low",
  },
  dashscope: {
    size: "1664*928",
    n: 1,
    negative_prompt: "",
    prompt_extend: true,
    watermark: false,
  },
};

// v0.3：各协议对应的 size 输入框占位提示（切换 protocol 时联动刷新）
const SIZE_PLACEHOLDERS = {
  openai: "例: 1024x1024 / 1536x1024 / auto；留空=用JSON",
  dashscope: "例: 1664*928 / 2048*2048；留空=用JSON",
};

// 统一 toast 入口；extensionManager 缺失时降级为 console.warn
function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) {
    t.add({ severity, summary, detail, life: 3000 });
  } else {
    console.warn(`[OpenAPI] ${severity}: ${summary}${detail ? " - " + detail : ""}`);
  }
}

// 按钮回调：拉取远端模型列表并回填 model 下拉选项
async function fetchModels(node, buttonWidget) {
  const find = (name) => node.widgets?.find((w) => w.name === name) ?? null;
  const baseUrl = (find("base_url")?.value ?? "").trim();
  if (!baseUrl) {
    toast("error", "请先填写 base_url");
    return;
  }
  const apiKey = find("api_key")?.value ?? "";
  // v0.2：读取 protocol COMBO（openai/dashscope）；旧工作流节点无此控件时默认 openai
  const protocol = find("protocol")?.value || "openai";
  const originalLabel = buttonWidget.label;
  buttonWidget.disabled = true;
  buttonWidget.label = "获取中…";
  try {
    const res = await api.fetchApi(ROUTE_FETCH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_url: baseUrl, api_key: apiKey, protocol: protocol }),
    });
    const json = await res.json();
    if (res.ok && json?.ok && Array.isArray(json.models)) {
      if (json.models.length > 0) {
        const modelWidget = find("model");
        if (modelWidget) {
          modelWidget.options = modelWidget.options || {};
          modelWidget.options.values = json.models;
          if (!json.models.includes(modelWidget.value)) {
            modelWidget.value = json.models[0];
          }
        }
        // 画布重绘: 新版前端 app.canvas 无 setDirtyCanvas (实机 QA 实测),
        // 以 app.graph.setDirtyCanvas 为主、canvas 为后备; 重绘失败绝不影响成功反馈
        try {
          if (app.graph && typeof app.graph.setDirtyCanvas === "function") {
            app.graph.setDirtyCanvas(true, true);
          } else if (app.canvas && typeof app.canvas.setDirtyCanvas === "function") {
            app.canvas.setDirtyCanvas(true, true);
          }
        } catch {
          // 重绘仅为视觉刷新, 失败静默忽略
        }
        toast("success", "模型列表已更新", `共 ${json.models.length} 个模型`);
      } else {
        toast("warn", "远端未返回任何模型");
      }
    } else {
      toast("error", "获取模型失败", json?.error ?? `HTTP ${res.status}`);
    }
  } catch (e) {
    toast("error", "获取模型失败", String(e?.message ?? e));
  } finally {
    buttonWidget.disabled = false;
    buttonWidget.label = originalLabel;
  }
}

// 按名称查找节点控件；widgets 缺失（异常节点状态）时返回 null
function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name) ?? null;
}

// v0.3.1：COMBO 中文标签。{content, value} 是 litegraph 原生 options 格式，
// 新旧两代前端均支持；控件值、工作流序列化、服务端 value_not_in_list 校验
// 全部仍为英文正典值——纯显示层映射，零向后兼容风险（旧工作流照常加载）。
const ZH_COMBO_LABELS = {
  "": "（留空不发送）",
  openai: "openai（兼容端点）",
  dashscope: "dashscope（阿里百炼原生）",
  auto: "auto（自动）",
  high: "high（高）",
  medium: "medium（中）",
  low: "low（低）",
  transparent: "transparent（透明）",
  opaque: "opaque（不透明）",
};

// 需要本地化的 5 个协议/基本参数下拉（model 是远端动态列表，不本地化）
const LOCALIZED_COMBOS = ["protocol", "quality", "output_format", "background", "moderation"];

// 把指定 combo 的 values 换成中文标签对象；已是对象形态（已本地化）则跳过，
// 无中文映射的值保持原样。任何失败静默（显示层绝不反噬节点功能）。
function localizeCombo(node, name) {
  try {
    const w = findWidget(node, name);
    const vals = w?.options?.values;
    if (!Array.isArray(vals) || !vals.every((v) => typeof v === "string")) return;
    const mapped = vals.map((v) => (ZH_COMBO_LABELS[v] ? { content: ZH_COMBO_LABELS[v], value: v } : v));
    if (mapped.some((m) => typeof m === "object")) w.options.values = mapped;
  } catch {
    /* 静默失败 */
  }
}

function localizeCombos(node) {
  for (const name of LOCALIZED_COMBOS) localizeCombo(node, name);
}

// “参数模板”按钮回调：按当前协议把模板 JSON 骨架写入 params 文本域。
// 旧工作流可能没有 params 控件——此时静默无操作；protocol 缺失时默认 openai（与 fetchModels 一致）
function applyParamsTemplate(node) {
  const paramsW = findWidget(node, "params");
  if (!paramsW) return;
  const protocol = findWidget(node, "protocol")?.value || "openai";
  const template = protocol === "dashscope" ? PARAMS_TEMPLATES.dashscope : PARAMS_TEMPLATES.openai;
  paramsW.value = JSON.stringify(template, null, 2);
  // 手动触发控件回调，让 litegraph 把新值落入 widgets_values 并随工作流持久化
  try {
    paramsW.callback?.(paramsW.value, app.canvas, node, paramsW);
  } catch {
    // 新旧前端控件回调签名不一，失败不影响已写入的值
  }
  toast("info", "已写入参数模板", "JSON 中同名键优先于交互控件；留空的交互控件不发送");
}

// 按协议刷新 size 输入框占位提示。旧工作流可能缺 size/protocol 控件；
// 占位符仅为视觉提示，任何异常一律静默吞掉，绝不影响节点功能。
function syncSizePlaceholder(node, protocolValue) {
  try {
    const sizeW = findWidget(node, "size");
    if (!sizeW) return;
    sizeW.options = sizeW.options || {};
    sizeW.options.placeholder =
      (protocolValue || "openai") === "dashscope" ? SIZE_PLACEHOLDERS.dashscope : SIZE_PLACEHOLDERS.openai;
  } catch {
    // 静默失败
  }
}

app.registerExtension({
  name: "ComfyUI.OpenAPI.FetchModels",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;

    // 节点创建后追加按钮 widget（v0.3.1：上游扩展异常已隔离，按钮必达）
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      // v0.3.1 修复：上游扩展（注册在本扩展之前的其他节点包）若在其 onNodeCreated
      // 中抛异常（如环境里抛 clipboard 错误的 LLM 包），曾一并带走本节点按钮。
      // 现在隔离为 console.error，保证下方按钮与本地化始终执行。
      let r;
      try {
        r = origOnNodeCreated?.apply(this, args);
      } catch (e) {
        console.error("[OpenAPI] 上游扩展的 onNodeCreated 抛出异常（已隔离，本节点按钮不受影响）:", e);
      }
      const w = this.addWidget("button", "获取模型列表", null, () => {
        fetchModels(this, w);
      });
      w.name = "Fetch Models"; // 供测试按 name 查找

      // v0.3：“参数模板”按钮——按当前协议把 params 文本域填充为 JSON 骨架
      const tplW = this.addWidget("button", "参数模板", null, () => {
        applyParamsTemplate(this);
      });
      tplW.name = "Params Template"; // 供测试按 name 查找

      // v0.3.1：protocol/基本参数下拉应用中文标签（值不变，纯显示层）
      localizeCombos(this);

      // v0.3：包装 protocol 控件回调——切换协议后联动刷新 size 占位提示。
      // v0.3.1：上游回调异常同样隔离（不再向外传播打断交互）
      const node = this;
      try {
        const protoW = findWidget(this, "protocol");
        if (protoW) {
          const origProtoCb = protoW.callback;
          protoW.callback = function (...cbArgs) {
            try {
              origProtoCb?.apply(this, cbArgs);
            } catch (e) {
              console.error("[OpenAPI] protocol 上游回调抛出异常（已隔离）:", e);
            }
            syncSizePlaceholder(node, cbArgs[0]);
          };
          syncSizePlaceholder(node, protoW.value); // 创建时按当前协议初始化占位符
        }
      } catch (e) {
        console.error("[OpenAPI] protocol 回调包装失败（不影响节点功能）:", e);
      }
      return r;
    };

    // 加载配置后按 base_url 从缓存预填 api_key（尽力而为，失败静默）
    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      // v0.3.1：同 onNodeCreated——隔离上游 onConfigure 异常，本钩子后续步骤必达
      let r;
      try {
        r = origOnConfigure?.apply(this, arguments);
      } catch (e) {
        console.error("[OpenAPI] 上游扩展的 onConfigure 抛出异常（已隔离）:", e);
      }
      // v0.3.1：litegraph 恢复 widgets_values 时不触发控件回调，且节点定义刷新
      // 会把 options.values 重置回后端纯字符串——故加载工作流后补做一次
      // 中文标签本地化 + 按还原 protocol 值刷新 size 占位符（内部均全静默）
      localizeCombos(this);
      syncSizePlaceholder(this, findWidget(this, "protocol")?.value);
      (async () => {
        try {
          const apiKeyW = this.widgets?.find((x) => x.name === "api_key");
          const baseUrlW = this.widgets?.find((x) => x.name === "base_url");
          const baseUrl = (baseUrlW?.value ?? "").trim();
          if (apiKeyW && (apiKeyW.value == null || apiKeyW.value === "") && baseUrl) {
            const res = await api.fetchApi(`${ROUTE_CACHE}?base_url=${encodeURIComponent(baseUrl)}`);
            const json = await res.json();
            if (json?.ok && json.api_key) apiKeyW.value = json.api_key;
          }
        } catch {
          /* 预填失败可忽略 */
        }
      })();
      return r;
    };
  },
});
