# THCUNews Classifier (Frontend)

这是一个基于 React + TypeScript + Vite 构建的现代化新闻文本分类系统前端界面。项目采用了 Hero Page 设计风格，集成了拖拽上传、智能交互与数据可视化功能。

## 🛠 技术栈

- **核心框架**: React 18, TypeScript, Vite
- **样式方案**: Tailwind CSS (极简原子化 CSS)
- **UI 组件**: 自定义 Glassmorphism (毛玻璃) 风格组件, Lucide React (图标)
- **工具库**: `clsx`, `tailwind-merge` (样式合并)
- **可视化**: Recharts (数据图表)

---

## 🚀 快速启动 (Quick Start)

如果您已经获取了本项目代码，请按以下步骤启动：

1.  **安装依赖**:
    ```bash
    npm install
    ```
2.  **启动开发服务器**:
    ```bash
    npm run dev
    ```
    打开浏览器访问 `http://localhost:3000` 即可预览。

3.  **构建生产版本**:
    ```bash
    npm run build
    ```

---

## 📖 进阶：如何将 Tailwind CSS 切换为本地安装 (Remove CDN)

目前 `index.html` 可能通过 CDN 引入 Tailwind。为了更好的开发体验（智能提示、自定义配置）和生产性能，建议将其下载到本地配置。

**请严格按照以下步骤操作：**

### 1. 安装开发依赖
在项目根目录下运行：
```bash
npm install -D tailwindcss postcss autoprefixer
```

### 2. 初始化配置文件
运行以下命令，这会自动创建 `tailwind.config.js` 和 `postcss.config.js`：
```bash
npx tailwindcss init -p
```

### 3. 修改配置路径
打开 `tailwind.config.js`，修改 `content` 数组，让 Tailwind 知道去扫描哪些文件：

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // 可以在此处保留项目自定义的动画配置
      animation: {
        blob: "blob 7s infinite",
        shimmer: "shimmer 2s infinite",
      },
      keyframes: {
        blob: {
          "0%": { transform: "translate(0px, 0px) scale(1)" },
          "33%": { transform: "translate(30px, -50px) scale(1.1)" },
          "66%": { transform: "translate(-20px, 20px) scale(0.9)" },
          "100%": { transform: "translate(0px, 0px) scale(1)" },
        },
        shimmer: {
          "from": { transform: "translateX(-100%)" },
          "to": { transform: "translateX(100%)" },
        }
      }
    },
  },
  plugins: [],
}
```

### 4. 创建全局样式入口
在 `src` 文件夹下新建一个 `index.css` 文件，内容如下：

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* 如果需要，可以将 index.html 中的自定义 CSS 移到这里 */
```

### 5. 在代码中引入 CSS
打开 `src/index.tsx`，在顶部添加导入：

```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css'; // <--- 新增这一行
// ...
```

### 6. 移除 CDN
打开 `index.html`，删除 `<script src="https://cdn.tailwindcss.com"></script>` 这一行。

完成以上步骤后，重启 `npm run dev`，您就拥有了完整的本地 Tailwind 环境！

---

## 🔌 核心代码修改：如何连接后端

当前前端使用 `src/services/mockApi.ts` 模拟数据返回。要对接真实的 Python (FastAPI/Flask) 后端，请按以下步骤修改代码。

### 1. 修改 API 服务文件
打开 `src/services/mockApi.ts`，将原有代码替换为真实的 `fetch` 请求：

```typescript
import { PredictResponse, ModelType } from '../types';

// 修改为您的真实后端地址
const API_URL = "http://localhost:8000/predict"; 

export const mockAnalyze = async (text: string, model: ModelType): Promise<PredictResponse> => {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      // 构造请求体，字段名需与后端接收模型一致
      body: JSON.stringify({
        text: text,
        model_name: model // 例如后端可能需要 'model_name' 字段
      }),
    });

    if (!response.ok) {
      throw new Error(`Server Error: ${response.status}`);
    }

    const data = await response.json();
    return data as PredictResponse; // 确保后端返回 JSON 结构符合前端类型定义

  } catch (error) {
    console.error("API Call Failed:", error);
    throw error; // 抛出错误，让前端 UI 显示错误提示
  }
};
```

### 2. 后端数据结构要求
为了适配前端图表，建议后端返回的 JSON 格式如下：

```json
{
  "category": "体育",
  "confidence": 0.98,
  "inference_time_ms": 45,
  "probabilities": [
    { "name": "体育", "value": 98.5 },
    { "name": "娱乐", "value": 1.2 },
    { "name": "其他", "value": 0.3 }
  ]
}
```

### 3. 处理跨域 (CORS)
由于前端运行在 `localhost:3000`，后端运行在 `localhost:8000`，请务必在 FastAPI 中配置 CORS：

```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # 允许前端域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📂 核心文件目录

```
src/
├── components/          
│   ├── InputSection.tsx  # [核心] 支持拖拽的文件输入区域
│   ├── ResultSection.tsx # [核心] Recharts 图表展示区
│   └── ui/               # 通用 UI 组件
├── services/            
│   └── mockApi.ts        # [修改点] API 请求逻辑
├── App.tsx               # 页面主入口，包含 Hero 动画逻辑
└── types.ts              # TS 类型定义
```