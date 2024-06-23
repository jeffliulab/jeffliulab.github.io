md文件：记录、备忘录等
mkdocs：把md文件生成为静态网页
material for mkdocs：mkdocs的一个主题
github：代码托管网站
github page：免费的静态网页，可以存放mkdocs

【常规流程】

1、将md文件放在一个大的文件夹内

2、使用mkdocs生成静态网页

3、将静态网页同步到github page

【安装教程】
1. 安装mkdocs：```pip install mkdocs```
1. 使用 mkdocs new 命令创建一个新的 MkDocs 项目：
```mkdocs new my-project```
```cd my-project```
这将在当前目录下创建一个名为 my-project 的文件夹，并在其中生成一些初始文件和文件夹。
1. 项目目录结构大致如下：
my-project/
    docs/
        index.md
    mkdocs.yml
docs/ 目录：存放所有的 Markdown 文件，即项目文档内容。
index.md 文件：默认的主页文档。
mkdocs.yml 文件：项目的配置文件。
1. 在项目目录中，运行以下命令启动 MkDocs 开发服务器：
```mkdocs serve```
这将启动一个本地开发服务器，通常会在 http://127.0.0.1:8000 提供实时预览。每当修改文档时，页面将自动刷新。
1. 配置项目
配置文件 mkdocs.yml 用于自定义站点的行为和外观。
1. 构建站点
当你完成了所有的文档编写，可以运行以下命令生成静态网站文件：
```mkdocs build```
生成的静态文件将位于 site 目录中。





【自动生成目录】

mkdocs-awesome-pages-plugin：
```pip install mkdocs-awesome-pages-plugin```

在 mkdocs.yml 文件中，添加 mkdocs-awesome-pages-plugin 插件配置：
```markdown
site_name: My Project
theme: readthedocs

plugins:
  - search
  - awesome-pages
```





