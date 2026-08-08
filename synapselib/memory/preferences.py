"""用户偏好（§6.5）：JSON 文件读写 + 合并更新。

- 存储：data/user_preferences.json（settings.preferences_file）
- 读：文件不存在 → 空偏好，不崩
- 写：update 合并后原子写盘（临时文件 + os.replace，防写一半损坏 JSON）
- 生命周期：长期记忆的一种——跨会话持久，重启不丢
"""
import json
import os
from pathlib import Path



class UserPreferences:
    """用户偏好（§6.5 MVP：显式反馈）。

    内部就是一个 dict（self._data）：读 = 取字典值，写 = 改字典 + 落盘。
    """
    def __init__(self, path: Path) -> None:
        """DI：传入 JSON 文件路径；文件不存在 → 空偏好（不崩）。"""
        self.path = path
        if not path.exists():
            self._data = {}
        else:
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def get(self, key: str, default=None):
        """读取偏好值；键不存在时返回 default。"""
        return self._data.get(key, default)

    def update(self, feedback: dict) -> None:
        """合并新反馈并写盘（只动传入的键，不覆盖其他偏好）。"""
        self._data.update(feedback)
        self._save()

    def to_dict(self) -> dict:
        """返回偏好副本（副本：调用方修改不影响内部状态）。"""
        return dict(self._data)

    def _save(self) -> None:
        """原子写盘：先写临时文件，os.replace 原子替换（防写一半损坏）。"""
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
