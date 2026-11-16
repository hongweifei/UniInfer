# UniInfer Desktop

面向普通用户的统一推理工作台。桌面端在进程内启动 loopback API，前端只通过 HTTP 访问本地运行时。

## 功能

- 对话：文本、多图输入、流式输出、停止、历史记录。
- 语音转写：WAV 拖放/选择、Whisper 转写、复制与下载。
- 图像生成：提示词、尺寸、步数、引导强度、随机种子。
- 顶部一级导航集成在自绘标题栏中；左侧二级分类直接切换子页面。
- 模型：在线搜索 ModelScope / Hugging Face、服务端分页、下载、格式/状态筛选、预加载/卸载、删除、设备、导入、重启。
- 下载完成后自动 inspect 并生成默认 manifest；模型卡可调整放置策略、优先设备、CPU 回退与上下文长度。
- 设置：主题、对话默认值、连接信息、运行时参数与本地数据，全部使用 GUI 表单。
- 模型运行配置保存后即时卸载旧会话，下一次请求按新放置/上下文加载，不需要重启。

## 启动

在仓库根目录运行：

```bash
cargo desktop
```

Windows 也可以双击仓库根目录的 `desktop.cmd`，或直接运行构建后的 exe。模型发现会依次检查当前目录、exe 所在目录及其父目录，因此从资源管理器启动也能找到仓库的 `models/`。

在线模型由用户搜索仓库后下载，不再内置固定模型清单；搜索按页请求，每页 10 条。默认扫描 `models/*/manifest*.toml`，下载目录在项目 `models/`。也可用环境变量覆盖：

```text
UNIINFER_MANIFESTS=path/a/manifest.toml;path/b/manifest.toml
UNIINFER_MANIFEST=path/manifest.toml
UNIINFER_CONFIG=path/uniinfer.toml
```

## 代码组织

```text
src/main.rs          Tauri 命令：状态、重启、窗口主题、目录选择
src/api_server.rs    进程内 API、配置与模型发现
src/model_store.rs   模型搜索、下载计划、models-cat 下载、inspect 与配置注册
ui/index.html        单页外壳
ui/styles/           tokens / base / shell / components / views
ui/js/main.js        bootstrap、路由、模型选择、shell
ui/js/api.js         HTTP/SSE/FormData 客户端
ui/js/state.js       持久化状态与对话
ui/js/dom.js         无框架 DOM 工具
ui/js/format.js      模态/模型/错误格式化
ui/js/icons.js       内联 SVG 图标
ui/js/components/    通用 UI（Toast、Modal、在线模型搜索）
ui/js/views/         对话、转写、图像、模型、设置
```

前端无构建步骤，使用原生 ES modules。新增视图只需在 `views/` 下实现 `mount(root, ctx)` 并注册到 `main.js`。
