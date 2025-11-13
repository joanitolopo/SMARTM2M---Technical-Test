"""
AUTOMATED Data Collection (all combinations) with confirmation
- Tries all 32 combinations of 5 components
- For each combination, optionally captures multiple angles
- Before saving each capture, asks user: "Save this screenshot? [y/n/q]"
- Hides UI overlays before taking canvas screenshot and restores afterwards
"""

import os
import time
import json
import random
import tempfile
import io
from itertools import combinations, product

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver import ActionChains
from selenium.common.exceptions import NoSuchElementException, WebDriverException

from PIL import Image
import numpy as np

class ImprovedCarDataCollector:
    def __init__(self, url, output_dir="car_dataset_cropped"):
        self.url = url
        self.output_dir = output_dir
        self.setup_directories()
        self.driver = None

        # components mapping (key -> button display text)
        self.components = {
            'front_left': 'Front Left Door',
            'front_right': 'Front Right Door',
            'rear_left': 'Rear Left Door',
            'rear_right': 'Rear Right Door',
            'hood': 'Hood'
        }

        self.current_state = {k: 'closed' for k in self.components.keys()}

    def setup_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels", exist_ok=True)
        print(f"✅ Directories created at: {self.output_dir}")

    def get_chromedriver_executable(self):
        path = ChromeDriverManager().install()
        if os.path.isfile(path) and path.lower().endswith(".exe"):
            return path
        base_dir = path if os.path.isdir(path) else os.path.dirname(path)
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if fname.lower().startswith("chromedriver") and (fname.lower().endswith(".exe") or fname.lower() == "chromedriver"):
                    return os.path.join(root, fname)
        raise FileNotFoundError(f"chromedriver executable not found inside {base_dir}")

    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        exe_path = self.get_chromedriver_executable()
        service = Service(exe_path)
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        time.sleep(1)
        print("✅ Chrome driver initialized with fixed window size")

    def open_page(self):
        self.driver.get(self.url)
        time.sleep(4)
        print(f"✅ Page loaded: {self.url}")

    def find_canvas_element(self):
        try:
            return self.driver.find_element(By.TAG_NAME, "canvas")
        except NoSuchElementException:
            return None

    def find_buttons(self):
        try:
            all_buttons = self.driver.find_elements(By.TAG_NAME, "button")
            buttons = {}
            for b in all_buttons:
                text = b.text.strip()
                if not text:
                    continue
                if "Front Left" in text:
                    buttons['front_left'] = b
                elif "Front Right" in text:
                    buttons['front_right'] = b
                elif "Rear Left" in text:
                    buttons['rear_left'] = b
                elif "Rear Right" in text:
                    buttons['rear_right'] = b
                elif "Hood" in text:
                    buttons['hood'] = b
            print(f"✅ Found {len(buttons)} control buttons")
            return buttons
        except Exception as e:
            print("❌ Error finding buttons:", e)
            return {}

    def click_button(self, button):
        try:
            button.click()
            time.sleep(1.5) 
            return True
        except Exception as e:
            try:
                
                self.driver.execute_script("arguments[0].click();", button)
                time.sleep(1.5)
                return True
            except Exception as e2:
                print("❌ Error clicking button:", e, e2)
                return False

    def rotate_view(self, direction='left'):
        canvas = self.find_canvas_element()
        if not canvas:
            return False
        action = ActionChains(self.driver)
        try:
            if direction == 'left':
                action.click_and_hold(canvas).move_by_offset(-150, 0).release().perform()
            elif direction == 'right':
                action.click_and_hold(canvas).move_by_offset(150, 0).release().perform()
            elif direction == 'up':
                action.click_and_hold(canvas).move_by_offset(0, -80).release().perform()
            elif direction == 'down':
                action.click_and_hold(canvas).move_by_offset(0, 80).release().perform()
            elif direction == 'diagonal_ne':
                action.click_and_hold(canvas).move_by_offset(100, -50).release().perform()
            elif direction == 'diagonal_se':
                action.click_and_hold(canvas).move_by_offset(100, 50).release().perform()
            time.sleep(0.8)
            return True
        except Exception as e:
            print("⚠️ rotate_view failed:", e)
            return False

    # JS helper to hide overlay siblings of canvas ancestor and restore
    HIDE_JS = r"""
    (function(){
      const canvas = document.querySelector('canvas');
      if(!canvas) return {ok:false, reason:'no-canvas'};
      // find ancestor likely grouping canvas & overlays
      let anc = canvas.parentElement;
      while(anc && anc !== document.body) {
        if (anc.childElementCount > 1) break;
        anc = anc.parentElement;
      }
      if(!anc) anc = canvas.parentElement || document.body;
      // hide non-canvas children
      const changed = [];
      for(const child of Array.from(anc.children)) {
        if (child.contains(canvas)) continue;
        child.setAttribute('data-old-display', child.style.display || '');
        child.setAttribute('data-old-visibility', child.style.visibility || '');
        child.setAttribute('data-old-pointer', child.style.pointerEvents || '');
        child.style.display = 'none';
        child.style.visibility = 'hidden';
        child.style.pointerEvents = 'none';
        changed.push(child);
      }
      // also hide absolute-positioned overlays inside anc
      const absCandidates = anc.querySelectorAll('*');
      for(const el of absCandidates) {
        if (el.contains(canvas)) continue;
        const cs = window.getComputedStyle(el);
        if (cs && cs.position === 'absolute') {
          el.setAttribute('data-old-display', el.style.display || '');
          el.setAttribute('data-old-visibility', el.style.visibility || '');
          el.setAttribute('data-old-pointer', el.style.pointerEvents || '');
          el.style.display = 'none';
          el.style.visibility = 'hidden';
          el.style.pointerEvents = 'none';
        }
      }
      return {ok:true, changed: changed.length};
    })();
    """
    RESTORE_JS = r"""
    (function(){
      const elems = document.querySelectorAll('[data-old-display]');
      for(const el of elems) {
        el.style.display = el.getAttribute('data-old-display') || '';
        el.style.visibility = el.getAttribute('data-old-visibility') || '';
        el.style.pointerEvents = el.getAttribute('data-old-pointer') || '';
        el.removeAttribute('data-old-display');
        el.removeAttribute('data-old-visibility');
        el.removeAttribute('data-old-pointer');
      }
      return {ok:true, restored: elems.length};
    })();
    """

    def get_canvas_image(self):
        """
        Return PIL.Image of the canvas area.
        Hide overlays first, then screenshot canvas, then restore.
        """
        canvas = self.find_canvas_element()
        if canvas:
            try:
                # hide overlays
                try:
                    self.driver.execute_script(self.HIDE_JS)
                except Exception:
                    pass
                time.sleep(0.05) 
                png = canvas.screenshot_as_png
                img = Image.open(io.BytesIO(png)).convert("RGB")
                # restore overlays
                try:
                    self.driver.execute_script(self.RESTORE_JS)
                except Exception:
                    pass
                return img
            except WebDriverException as e:
                print("⚠️ canvas.screenshot failed, falling back to full screenshot:", e)
                try:
                    self.driver.execute_script(self.RESTORE_JS)
                except Exception:
                    pass
        try:
            png = self.driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(png)).convert("RGB")
            w,h = img.size
            left = int(w * 0.15)
            top = int(h * 0.12)
            right = int(w * 0.85)
            bottom = int(h * 0.9)
            crop = img.crop((left, top, right, bottom))
            return crop
        except Exception as e:
            print("❌ get_canvas_image fallback failed:", e)
            return None

    def capture_screenshot_with_confirmation(self, filename, state, confirm=True):
        """
        Capture screenshot (canvas), but ask user for confirmation before saving.
        confirm=True -> ask prompt; if False -> auto-save
        Returns True if saved.
        """
        img = self.get_canvas_image()
        if img is None:
            print("❌ Could not capture image for", filename)
            return False

        # Resize to training size
        img_resized = img.resize((640, 480), Image.Resampling.LANCZOS)

        if confirm:
            ans = input(f"Save screenshot '{filename}'? [y/n/q]: ").strip().lower()
            if ans in ('q', 'quit'):
                print("🛑 User requested quit.")
                return "quit"
            if ans not in ('y', 'yes'):
                print("⤷ Skipped:", filename)
                return False

        # Save image and label
        img_path = f"{self.output_dir}/images/{filename}.png"
        label_path = f"{self.output_dir}/labels/{filename}.json"
        try:
            img_resized.save(img_path)
            with open(label_path, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"  ✓ Saved: {filename}")
            return True
        except Exception as e:
            print("❌ Error saving image/label:", e)
            return False

    def apply_state(self, target_state, buttons):
        """
        Change the page state (open/closed) to match target_state by clicking necessary buttons.
        Returns after actions completed.
        """
        for key, desired in target_state.items():
            if key not in buttons:
                continue
            if self.current_state.get(key, 'closed') != desired:
                # click to toggle
                ok = self.click_button(buttons[key])
                if ok:
                    self.current_state[key] = desired
                    time.sleep(0.6)
                else:
                    print(f"⚠️ Failed toggling {key}")

    def all_combinations(self):
        """Generator for all binary combos of 5 components as dicts"""
        keys = list(self.components.keys())
        for bits in product([0,1], repeat=len(keys)):
            state = {}
            for k, b in zip(keys, bits):
                state[k] = 'open' if b==1 else 'closed'
            yield state

    def run_all_combinations_quick(self, confirm_each=True, angles_per_combo=3):
        """
        Automate through all 32 combinations.
        confirm_each: if True -> before saving each screenshot ask user.
        angles_per_combo: number of different rotations per combo to capture.
        """
        try:
            self.setup_driver()
            self.open_page()
            buttons = self.find_buttons()
            if not buttons:
                print("❌ No control buttons found. Exiting.")
                return

            # ensure baseline closed: click any opened buttons (best-effort)
            for k, b in buttons.items():
                pass

            total_saved = 0
            combo_idx = 0
            for state in self.all_combinations():
                combo_idx += 1
                print("\n" + "="*60)
                print(f"COMBO {combo_idx:02d}/32 -> {state}")
   
                for k in list(self.current_state.keys()):
                    if self.current_state[k] == 'open' and k in buttons:
                        self.click_button(buttons[k])
                        self.current_state[k] = 'closed'
                        time.sleep(0.3)

                for k, v in state.items():
                    if v == 'open' and k in buttons:
                        self.click_button(buttons[k])
                        self.current_state[k] = 'open'
                        time.sleep(0.4)

                directions = ['left', 'right', 'up', 'diagonal_ne', 'diagonal_se']
                for angle_idx in range(angles_per_combo):
                    dir_choice = random.choice(directions)
                    self.rotate_view(dir_choice)
                    time.sleep(0.5)

                    filename = f"combo_{combo_idx:02d}_{'_'.join([k[0] for k,v in state.items() if v=='open'])}_angle{angle_idx}"
                    filename = filename.replace(' ', '_')[:120]

                    res = self.capture_screenshot_with_confirmation(filename, state, confirm=confirm_each)
                    if res == "quit":
                        print("User requested stop. Exiting collection loop.")
                        return total_saved
                    if res:
                        total_saved += 1
                time.sleep(0.3)

            print("\n" + "="*60)
            print(f"All combos processed. Saved {total_saved} images.")
            return total_saved

        finally:
            if self.driver:
                print("Closing browser...")
                time.sleep(1.2)
                self.driver.quit()

def main():
    URL = "https://euphonious-concha-ab5c5d.netlify.app/"
    OUTPUT_DIR = "car_dataset_cropped"

    print("AUTOMATED ALL-COMBINATIONS DATA COLLECTION")
    print("="*60)
    print(f"Target URL: {URL}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("Note: The script will ask confirmation before saving each screenshot.")
    print("="*60)

    collector = ImprovedCarDataCollector(URL, OUTPUT_DIR)

    # Ask user before starting
    go = input("Start automated collection? [y/n]: ").strip().lower()
    if go not in ('y','yes'):
        print("Aborted by user.")
        return

    saved = collector.run_all_combinations_quick(confirm_each=False, angles_per_combo=5)

    print("\nDone. Total saved images:", saved)
    print("Check folder:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
