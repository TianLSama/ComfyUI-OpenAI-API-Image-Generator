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

// 包装 protocol 控件回调（幂等：控件上的标记防重复包装）。
// 上游回调异常被隔离为 console.error——其他扩展崩不了本节点的占位符联动。
function wrapProtocolCallback(node) {
  const protoW = findWidget(node, "protocol");
  if (!protoW || protoW.__openapiProtoWrapped) return;
  const origProtoCb = protoW.callback;
  protoW.callback = function (...cbArgs) {
    try {
      origProtoCb?.apply(this, cbArgs);
    } catch (e) {
      console.error("[OpenAPI] protocol 上游回调抛出异常（已隔离）:", e);
    }
    syncSizePlaceholder(node, cbArgs[0]);
  };
  protoW.__openapiProtoWrapped = true;
}

// v0.3.1 核心修复：幂等地保证「按钮 + 中文标签 + 占位符联动」就位。
// 背景：按钮是动态注入 widget，不在后端节点定义里——新版前端在节点定义
// 变化（本插件 7→14 控件升级即触发）时会重建 node.widgets 且不再执行
// onNodeCreated，动态按钮随重建丢失。对策：挂全生命周期（onNodeCreated /
// onConfigure / updateNodeData / graphNodeMounted），每次按 name 检测缺失
// 才补注，重复调用无副作用；addWidget 追加在 widgets 末尾，不打乱
// widgets_values 的索引映射。任何失败只记日志，绝不向上抛。
function ensureWidgets(node) {
  try {
    if (!node?.addWidget) return;
    if (!findWidget(node, "Fetch Models")) {
      const w = node.addWidget("button", "获取模型列表", null, () => {
        fetchModels(node, w);
      });
      w.name = "Fetch Models"; // 供测试按 name 查找
    }
    if (!findWidget(node, "Params Template")) {
      node.addWidget("button", "参数模板", null, () => {
        applyParamsTemplate(node);
      }).name = "Params Template"; // 供测试按 name 查找
    }
    localizeCombos(node);
    wrapProtocolCallback(node);
    syncSizePlaceholder(node, findWidget(node, "protocol")?.value);
  } catch (e) {
    console.error("[OpenAPI] ensureWidgets 失败（按钮可能缺失，请报告此日志）:", e);
  }
}

app.registerExtension({
  name: "ComfyUI.OpenAPI.FetchModels",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;

    // 节点创建：先执行上游链（异常隔离），再幂等注入本扩展 widget
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      let r;
      try {
        r = origOnNodeCreated?.apply(this, args);
      } catch (e) {
        console.error("[OpenAPI] 上游扩展的 onNodeCreated 抛出异常（已隔离）:", e);
      }
      ensureWidgets(this);
      return r;
    };

    // 工作流加载恢复：上游异常隔离后补注入（litegraph 恢复 widgets_values 不触发
    // 控件回调，protocol/size 联动需要在 configure 后重建）
    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      let r;
      try {
        r = origOnConfigure?.apply(this, arguments);
      } catch (e) {
        console.error("[OpenAPI] 上游扩展的 onConfigure 抛出异常（已隔离）:", e);
      }
      ensureWidgets(this);
      // 加载配置后按 base_url 从缓存预填 api_key（尽力而为，失败静默）
      (async () => {
        try {
          const apiKeyW = findWidget(this, "api_key");
          const baseUrl = (findWidget(this, "base_url")?.value ?? "").trim();
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

    // ★ v0.3.1 关键补漏：新版前端在节点定义变化（如本插件 7→14 控件升级）时会
    // 调 updateNodeData 重建全部 widgets 且【不再执行 onNodeCreated】——动态注入
    // 的按钮正是于此丢失（用户报告的「按钮消失」根因）。在其后幂等补注。
    const origUpdateNodeData = nodeType.prototype.updateNodeData;
    if (typeof origUpdateNodeData === "function") {
      nodeType.prototype.updateNodeData = function (...args) {
        let r;
        try {
          r = origUpdateNodeData.apply(this, args);
        } catch (e) {
          console.error("[OpenAPI] 上游 updateNodeData 抛出异常（已隔离）:", e);
          throw e; // 维持原语义：重建失败本就该向外暴露
        }
        ensureWidgets(this);
        return r;
      };
    }
  },

  // 新前端（Vue）每个节点 DOM 挂载后的钩子——最后防线：无论此前哪个环节
  // 重建/清掉了按钮，挂载完成后都能补回来。旧前端无此钩子，自然降级。
  graphNodeMounted(litegraphNode) {
    if (litegraphNode?.type === NODE_CLASS) ensureWidgets(litegraphNode);
  },
});
