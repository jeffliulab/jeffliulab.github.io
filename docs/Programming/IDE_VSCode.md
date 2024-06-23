#### Terminal
code xxx   | 创建新文件
touch xxx  | 创建新文件（不打开）
cd ..      | 返回上一级
ls         | 查看文件列表
mkdir xxx  | 创建目录
mkdir -p xxx/xxx | 创建目录（自动补全不存在的中间目录）


#### 在vscode中运行C文件

| 指令  | 在 Terminal 中输入 |
|-------|---------------------|
| Write | `code xxx.c` |
| Compile | `make xxx` |
| Run | `./xxx` |

Write指令中的code用法请注意：首次运行 VSCode 时，需要打开命令面板（使用快捷键 F1 或 Ctrl+Shift+P），然后搜索并执行 Shell Command: Install 'code' command in PATH 命令来配置 code 命令。

#### 运行java文件

1. 前置条件：安装java extension pack和JDK。
1. 创建项目：`cmd+shift+p`，选择`java: create java project`
1. 运行文件：
    1. 使用 javac 命令编译 Java 文件。这将生成一个 .class 文件，其中包含 Java 字节码。
    1. 输入：`javac HelloWorld.java`
    1. 成功编译后，你会在同一目录中看到一个名为 HelloWorld.class 的文件。使用 java 命令运行编译后的 Java 类：`java HellowWorld`

(1) create a java project:
    mkdir MyJavaProject
    cd MyJavaProject
    mkdir -p src/main/java/com/example
    touch src/main/java/com/example/Main.java
(2) write java file
(3) compile:
    javac src/main/java/com/example/Main.java
    java -cp src/main/java com.example.Main


## Python for .py
compile: 
    python3 xxx.py
REPL: (Read-Eval-Print Loop)
    python3
    # ...instructions
    # quit() or exit() to quit
    # Ctrl-d to stop