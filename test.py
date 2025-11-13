from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import os

def get_chromedriver_executable():
    path = ChromeDriverManager().install()
    if os.path.isfile(path) and path.lower().endswith(".exe"):
        return path

    base_dir = path if os.path.isdir(path) else os.path.dirname(path)

    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.lower().startswith("chromedriver") and (fname.lower().endswith(".exe") or fname.lower() == "chromedriver"):
                return os.path.join(root, fname)

    raise FileNotFoundError(f"chromedriver executable not found inside {base_dir}. Install failed or archive malformed.")

# Usage inside setup_driver
exe_path = get_chromedriver_executable()
# service = Service(exe_path)
# self.driver = webdriver.Chrome(service=service, options=chrome_options)
