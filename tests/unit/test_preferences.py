"""memory/preferences.py 单元测试：空文件兜底、合并更新、UTF-8 持久化、原子写盘。

覆盖验收标准（docs/M6_memory_design.md §5 preferences 节）+ 用户修过的两个 bug：
- self.path 必须存（否则 _save 的 AttributeError 回归）
- 读写都要 encoding="utf-8"（否则中文 UnicodeDecodeError 回归）
"""
from __future__ import annotations

from synapselib.memory.preferences import UserPreferences


class TestInit:
    def test_missing_file_empty_prefs(self, tmp_path):
        """文件不存在 → 空偏好，不崩。"""
        p = UserPreferences(tmp_path / "prefs.json")
        assert p.to_dict() == {}

    def test_get_returns_default_when_missing(self, tmp_path):
        p = UserPreferences(tmp_path / "prefs.json")
        assert p.get("depth_preference") is None
        assert p.get("depth_preference", "浅") == "浅"


class TestUpdate:
    def test_update_merges_not_overwrites(self, tmp_path):
        """update 合并不覆盖其他键。"""
        p = UserPreferences(tmp_path / "prefs.json")
        p.update({"a": 1})
        p.update({"b": 2})
        assert p.to_dict() == {"a": 1, "b": 2}

    def test_persists_across_instances(self, tmp_path):
        """写盘后可重新读回（持久化 = 跨会话不丢）。"""
        path = tmp_path / "prefs.json"
        UserPreferences(path).update({"depth_preference": "深入", "report_style": "markdown"})

        reopened = UserPreferences(path)  # 模拟重启后重新加载
        assert reopened.get("depth_preference") == "深入"
        assert reopened.get("report_style") == "markdown"

    def test_chinese_utf8_roundtrip(self, tmp_path):
        """编码坑回归：写盘 ensure_ascii=False，读盘同编码，中文无损。"""
        path = tmp_path / "prefs.json"
        UserPreferences(path).update({"research_interests": ["大模型", "检索"]})

        assert "大模型" in path.read_text(encoding="utf-8")  # 文件里是可读中文，非 \u 转义
        assert UserPreferences(path).get("research_interests") == ["大模型", "检索"]


class TestSave:
    def test_no_tmp_file_left_behind(self, tmp_path):
        """原子写盘收尾干净：os.replace 后无 .tmp 残留。"""
        path = tmp_path / "prefs.json"
        p = UserPreferences(path)
        p.update({"a": 1})
        assert not (tmp_path / "prefs.json.tmp").exists()

    def test_to_dict_returns_copy(self, tmp_path):
        """to_dict 返回副本：调用方修改不影响内部状态。"""
        p = UserPreferences(tmp_path / "prefs.json")
        p.update({"a": 1})
        d = p.to_dict()
        d["a"] = 999
        assert p.get("a") == 1