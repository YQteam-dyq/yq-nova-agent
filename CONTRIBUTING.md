# Contributing

感谢你愿意为这个项目贡献力量。

## 提交 PR

欢迎直接向 main 分支提交 PR。仓库开启了分支保护（Ruleset）：普通分支的 PR 需要 CI 通过后才能合并；仓库管理员可以绕过保护直接推送。

## 开发流程

- fork 或 clone 后先安装依赖，并跑通现有测试
- 修改代码，并为改动补充或更新测试
- 提交前运行 lint / typecheck / test（具体命令见 README 或 .github/workflows 下的 CI 配置）
- 推送分支并创建 PR，PR 会自动触发 CI 与 Sourcery 机器人审查

## 提交信息规范

推荐使用 Conventional Commits 风格，例如 feat / fix / refactor / docs / chore / ci，便于自动生成变更记录。

## 发布与维护

维护者合并 PR 使用 Squash 合并；合并后源分支会自动删除。版本发布按需打 tag。
