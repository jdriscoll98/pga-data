from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json
import time

def scrape_tournament_stats(url):
    # Setup Chrome driver with options
    options = webdriver.ChromeOptions()
    # Add window size to make sure everything is visible
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)
    
    try:
        print(f"Navigating to {url}")
        driver.get(url)
        leaderboard_data = []
        
        # Add a longer initial wait to let the page fully load
        time.sleep(5)
        print("Waiting for table to load...")
        
        # Wait for the leaderboard table to load
        table = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.chakra-table.css-r5jdff > tbody"))
        )
        print("Table found")
        
        # Find all player rows
        rows = table.find_elements(By.TAG_NAME, "tr")
        print(f"Found {len(rows)} player rows")
        
        for row in rows:
            try:
                # Check finish position first
                finish = row.find_element(By.CSS_SELECTOR, "td:nth-child(1)").text.strip()
                if finish == 'CUT':
                    print("Skipping CUT player")
                    continue
                
                # Get player name
                name_cell = row.find_element(By.CSS_SELECTOR, "td:nth-child(3)")
                name = name_cell.text.strip()
                print(f"\nProcessing player: {name} (Finish: {finish})")
                
                print("Clicking row...")
                row.click()
                print("Row clicked")
                
                time.sleep(2)  # Increased wait time after click
                
                print("Looking for stats tab...")
                # Try multiple possible tab selectors
                tab_selectors = [
                    "button[aria-label='Stats']",  # Most specific and stable
                    "button[role='tab'][data-index='1']",  # Alternative using role and index
                    ".chakra-tabs__tab[data-index='1']",  # Using class and index
                    "button.chakra-tabs__tab[aria-selected='false']"  # Using class
                ]
                
                stats_tab = None
                for selector in tab_selectors:
                    try:
                        stats_tab = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        print(f"Found stats tab with selector: {selector}")
                        break
                    except:
                        continue
                
                if stats_tab is None:
                    print("Could not find stats tab!")
                    # Take a screenshot for debugging
                    driver.save_screenshot("debug_screenshot.png")
                    raise Exception("Stats tab not found")
                
                stats_tab.click()
                print("Stats tab clicked")
                time.sleep(2)
                
                # Find all stat containers
                stat_containers = driver.find_elements(By.CSS_SELECTOR, "div.css-gg4vpm")
                
                stats = {}
                for container in stat_containers[1:]:
                    spans = container.find_elements(By.TAG_NAME, "span")
                    if len(spans) >= 3:
                        stat_name = spans[0].text.strip()
                        stat_value = spans[1].text.strip()
                        # Convert stat_value to float if possible
                        try:
                            stat_value = float(stat_value)
                        except ValueError:
                            pass
                        stats[stat_name] = stat_value
                
                leaderboard_data.append({
                    "name": name,
                    "finish": finish,
                    "stats": stats
                })

                # Close the all details panel
                close_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Close']")
                close_button.click()
                time.sleep(0.5)  # Wait for panel to close
                
            except Exception as e:
                print(f"Error processing player {name if 'name' in locals() else 'unknown'}: {str(e)}")
                continue
        
        return {"leaderboard": leaderboard_data}
    
    finally:
        driver.quit()

def main():
    tournaments = [
        "https://www.pgatour.com/tournaments/2022/mexico-open-at-vidanta/R2022540/leaderboard",
        "https://www.pgatour.com/tournaments/2023/mexico-open-at-vidanta/R2023540/leaderboard",
        "https://www.pgatour.com/tournaments/2024/mexico-open-at-vidanta/R2024540/leaderboard",
    ]
    
    all_tournament_data = {}
    
    for i, url in enumerate(tournaments, 2022):
        print(f"Scraping {i} tournament data...")
        tournament_data = scrape_tournament_stats(url)
        all_tournament_data[str(i)] = tournament_data
    
    # Save to JSON file
    with open('tournament_stats.json', 'w') as f:
        json.dump(all_tournament_data, f, indent=2)
    
    print("Data collection complete. Results saved to tournament_stats.json")

if __name__ == "__main__":
    main()
