# Standard Library
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NoReturn

# Third Party Library
from tqdm import tqdm


class DataCollector(ABC):
    """EEGデータを収集するための抽象基底クラス。"""

    def __init__(self, save_dir_path: str) -> None:
        """DataCollectorの初期化メソッドです。"""
        self.save_dir_path: str = save_dir_path
        self.__validate_save_dir_path()

    def __validate_save_dir_path(self) -> None:
        """
        save_dir_pathが有効なディレクトリパスかどうかを検証します。
        不適切な場合は例外を発生させます。
        """
        path: Path = Path(self.save_dir_path)
        if not path.exists():
            os.makedirs(path)
        elif not path.is_dir():
            raise ValueError(
                f"Provided path is not a directory: {self.save_dir_path}"
            )

    @abstractmethod
    def search_eeg_data(self) -> None:
        """EEGデータを検索し、結果を返す抽象メソッドです。"""


class URLDataCollector(DataCollector):
    """URLからEEGデータを収集するクラスです。"""

    def __init__(self, save_dir_path: str) -> None:
        """URLDataCollectorの初期化メソッドです。"""
        super().__init__(save_dir_path)

    def download_eeg_data(self, url: str) -> NoReturn:
        """指定されたURLからEEGデータをダウンロードし、保存します。
        ダウンロードが失敗した場合は例外を発生させます。
        """
        # URLからファイル名を抽出する
        filename = url.split("/")[-1]
        full_path = os.path.join(self.save_dir_path, filename)

        # save_dir_path がディレクトリであることを確認し、存在しない場合は作成する
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with requests.get(url, stream=True, timeout=10) as response:
            if response.status_code == requests.codes.ok:
                total_size_in_bytes = int(
                    response.headers.get("content-length", 0)
                )
                block_size = 1024  # 1 Kibibyte

                progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )

                with open(full_path, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
            else:
                raise Exception("Failed to download the EEG data.")


if __name__ == "__main__":
    url_data_collector = URLDataCollector(
        save_dir_path="/home/iplslam/EEG_Classification/data/row"
    )
    url_data_collector.download_eeg_data(
        "https://figshare.com/ndownloader/files/36324114"
    )
