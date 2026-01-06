# Java AI Integration Starter

基于 Spring AI 和 Vaadin 的智能对话系统。

## 功能特性

- ✅ Spring AI 集成（支持 OpenAI）
- ✅ Vaadin 现代化 UI
- ✅ 流式响应支持
- ✅ REST API 接口
- ✅ 分层架构设计（Service/Controller/View 分离）

## 快速开始

### 前置要求

- Java 17 或更高版本
- Maven（或使用项目自带的 Maven Wrapper）

### 配置 API Key

**重要：为了安全，API key 不会提交到 Git 仓库。**

#### 推荐方式：使用 secrets 配置文件

**优点：** 简单易用，无需设置环境变量，一次配置永久使用，配置文件已在 `.gitignore` 中，安全可靠。

1. **复制示例配置文件：**
   ```bash
   # Windows PowerShell
   Copy-Item src\main\resources\application-secrets.yaml.example src\main\resources\application-secrets.yaml
   
   # Windows CMD
   copy src\main\resources\application-secrets.yaml.example src\main\resources\application-secrets.yaml
   
   # Linux/Mac
   cp src/main/resources/application-secrets.yaml.example src/main/resources/application-secrets.yaml
   ```

2. **编辑 `application-secrets.yaml`，填入您的真实 API key：**
   ```yaml
   spring:
     ai:
       openai:
         api-key: sk-proj-your-actual-api-key-here
   ```

3. **直接运行应用即可**（Spring Boot 会自动加载此配置文件）

**注意：** `application-secrets.yaml` 已在 `.gitignore` 中，不会被提交到 Git。

#### 其他配置方式（可选）

**方式 1：使用环境变量**

**Windows PowerShell:**
```powershell
# 临时设置（当前会话有效）
$env:SPRING_AI_OPENAI_API_KEY="your-api-key-here"

# 永久设置
[System.Environment]::SetEnvironmentVariable('SPRING_AI_OPENAI_API_KEY', 'your-api-key-here', 'User')
```

**Windows CMD:**
```cmd
set SPRING_AI_OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
# 临时设置
export SPRING_AI_OPENAI_API_KEY="your-api-key-here"

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export SPRING_AI_OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**方式 2：使用系统属性**
```bash
mvn spring-boot:run -Dspring.ai.openai.api-key=your-api-key-here
```

#### 获取 OpenAI API Key

1. 访问 https://platform.openai.com/api-keys
2. 登录您的账户
3. 创建新的 API key
4. 复制并安全保存（只显示一次）

#### 安全提示

- ✅ **推荐：** 使用 secrets 配置文件或环境变量
- ❌ **不推荐：** 直接在 `application.yaml` 中硬编码 API key
- ⚠️ **警告：** 如果配置错误，应用启动时会在日志中显示相关错误信息

### 运行项目

#### 方式 1：使用运行脚本（最简单，推荐）

**PowerShell:**
```powershell
.\run.ps1
```

**CMD/Batch:**
```cmd
run.bat
```

这些脚本会自动设置 JAVA_HOME 并启动应用。

#### 方式 2：手动设置 JAVA_HOME 后使用 Maven Wrapper

**PowerShell:**
```powershell
# 查找并设置 JAVA_HOME
$javaPath = (Get-Command java).Source
$javaHome = Split-Path (Split-Path $javaPath)
$env:JAVA_HOME = $javaHome

# 运行项目
.\mvnw.cmd spring-boot:run
```

**CMD:**
```cmd
REM 需要先手动设置 JAVA_HOME 环境变量
set JAVA_HOME=C:\Program Files\Java\jdk-17
.\mvnw.cmd spring-boot:run
```

#### 方式 3：使用 Chocolatey 安装 Maven（如果已安装 Chocolatey）

```powershell
choco install maven
mvn spring-boot:run
```

#### 方式 4：使用 IDE

- **IntelliJ IDEA**：右键 `AiIntegrationApplication.java` -> Run
- **Eclipse**：右键项目 -> Run As -> Spring Boot App
- **VS Code**：安装 Spring Boot Extension Pack，然后运行

### 访问应用

- **Vaadin UI**: http://localhost:8080
- **REST API**: 
  - `GET /api/ai/chat?prompt=你好` - 同步调用
  - `POST /api/ai/stream` - 流式调用（SSE）

## 项目结构

```
src/main/java/com/example/ai/
├── AiIntegrationApplication.java    # 主应用类
├── service/
│   └── ChatService.java              # AI 业务逻辑层
├── controller/
│   └── AiChatController.java        # REST API 控制器
└── views/
    └── ChatView.java                 # Vaadin UI 视图
```

## 架构说明

项目采用分层架构设计：

- **ChatService**: 封装所有 Spring AI 逻辑，与 UI 层解耦
- **ChatView**: Vaadin UI 层，仅与 ChatService 交互
- **AiChatController**: REST API 层，同样使用 ChatService

这种设计使得：
- 可以轻松切换 UI 框架（Vaadin → React/Vue 等）
- 业务逻辑可复用
- 易于测试和维护

## 技术栈

- Spring Boot 3.2.0
- Spring AI 1.0.0-M4
- Vaadin 24.4.0
- Reactor (响应式编程)
