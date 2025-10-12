要在 **离线环境** 中将一个 **已经通过 npm 安装好依赖的 Node.js 项目** 完整迁移到另一台 **离线机器**，并确保能 **从本地缓存安装依赖**，你需要做以下几步：

---

### ✅ 目标
> 在 **没有网络** 的机器上，使用 **本地已有的 npm 包缓存**，完整还原项目依赖。

---

### ✅ 步骤概览

| 步骤 | 描述 |
|------|------|
| 1 | 在联网机器上打包项目 + 缓存 |
| 2 | 将缓存和项目一起拷贝到离线机器 |
| 3 | 在离线机器上配置 npm 使用本地缓存 |
| 4 | 安装依赖（离线） |

---

### ✅ 详细操作步骤

#### ✅ 1. 在联网机器上准备缓存

```bash
# 安装依赖（如果还没装）
npm install

# 创建一个本地缓存目录（比如叫 npm-cache-offline）
mkdir npm-cache-offline

# 将当前项目的依赖缓存到本地目录
npm cache verify
npm pack $(ls node_modules | sed 's/^/@/' | tr '\n' ' ')
```

> ⚠️ 注意：`npm pack` 会下载 `.tgz` 文件，但你可以跳过这一步，**直接拷贝缓存目录** 更稳妥。

---

#### ✅ 2. 拷贝缓存目录 + 项目

```bash
# 找到 npm 缓存目录
npm config get cache
# 通常是：~/.npm

# 拷贝缓存目录 + 项目源码
cp -r ~/.npm ./npm-cache-offline
cp -r your-project ./your-project
```

最终你得到两个目录：

- `npm-cache-offline`：包含所有 `.tgz` 包
- `your-project`：包含 `package.json` 和 `package-lock.json`

---

#### ✅ 3. 在离线机器上配置 npm 使用本地缓存

```bash
# 将缓存目录复制到离线机器
# 比如放到 /opt/npm-cache-offline

# 配置 npm 使用本地缓存
npm config set cache /opt/npm-cache-offline --global
npm config set registry file:///opt/npm-cache-offline --global
```

---

#### ✅ 4. 安装依赖（离线）

```bash
cd your-project
npm ci --offline
```

> ✅ `--offline` 参数强制 npm 使用本地缓存，不访问网络。

---

### ✅ 可选：使用 `npm-offline-registry`（更稳定）

如果你希望更“像 registry”的方式，可以用：

- [verdaccio](https://verdaccio.org/)（轻量私有 npm 仓库）
- 或 [npm-offline-registry](https://github.com/nolanlawson/npm-offline-registry)

#### 示例：用 verdaccio 做离线 registry

```bash
# 在联网机器上
npm install -g verdaccio
verdaccio &
npm set registry http://localhost:4873
npm publish --registry http://localhost:4873
```

然后打包 verdaccio 的存储目录（`~/.config/verdaccio/storage`）到离线机器，启动 verdaccio，离线机器指向它即可。

---

### ✅ 总结一句话

> **拷贝 `~/.npm` 缓存目录 + 项目源码 → 在离线机器上设置 `npm config set cache` 和 `registry` → 使用 `npm ci --offline` 安装依赖。**

---

### ✅ 附加建议

- 使用 `npm ci` 而不是 `npm install`，更快、更一致。
- 保留 `package-lock.json`，确保版本一致。
- 如果项目依赖 native 模块（如 node-sass、sqlite3），确保离线机器有编译环境（node-gyp、python、make）。

---

需要我帮你写一个一键打包脚本吗？