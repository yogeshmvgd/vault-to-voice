from vault_to_voice.config import Settings
from vault_to_voice.vault.obsidian import ObsidianVaultReader
from vault_to_voice.vault.reader import FilesystemVaultReader, VaultReader


def make_reader(settings: Settings) -> VaultReader:
    if settings.vault_format == "flat":
        return FilesystemVaultReader(settings.vault_path)
    return ObsidianVaultReader(
        settings.vault_path,
        daily_folder=settings.daily_folder,
        date_format=settings.daily_date_format,
        categories=settings.categories,
    )
