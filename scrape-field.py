from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import json
from datetime import datetime

# Set up Chrome options for headless mode
options = Options()
options.add_argument("--headless")

# Initialize the WebDriver
driver = webdriver.Chrome(options=options)
driver.set_page_load_timeout(30)  # Set page load timeout

try:
    driver.get("https://www.pgatour.com/tournaments/2025/mexico-open-at-vidantaworld/R2025540")

    # Get all player elements
    golfer_elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((
            By.CSS_SELECTOR, 
            "tr[class*='player-'][class*='css-'] td.css-1y9jg86 a"
        ))
    )

    # Store initial player data to avoid stale elements
    initial_player_data = [
        (elem.text.strip(), elem.get_attribute('href').split('?')[0])
        for elem in golfer_elements
        if elem.text.strip()
    ]

    # Create a dictionary to store all player data
    players_data = {}

    # Process each player
    for name, profile_url in initial_player_data:
        print(f"Processing player: {name}")
        stats_url = f"{profile_url}/stats"
        players_data[name] = {
            'profile_url': profile_url,
            'stats': {}
        }
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Navigate to stats page
                driver.get(stats_url)
                time.sleep(5)  # Increased wait time

                # Wait for and find the stats table
                table = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR, 
                        "table.chakra-table"
                    ))
                )

                # Find all stat rows in the table
                stat_rows = table.find_elements(By.CSS_SELECTOR, "tbody tr.css-79elbk")

                # Extract stats from each row
                for row in stat_rows:
                    try:
                        stat_name = row.find_element(By.CSS_SELECTOR, "td.css-1y9jg86 a .css-h2dnm1").text.replace(" >", "").strip()
                        value = row.find_element(By.CSS_SELECTOR, "td:nth-child(2) .css-1psnea4").text.strip()
                        rank = row.find_element(By.CSS_SELECTOR, "td:nth-child(3) .css-1psnea4").text.strip()
                        
                        players_data[name]['stats'][stat_name] = {
                            'value': value,
                            'rank': rank
                        }
                    except Exception as e:
                        print(f"Error extracting stat for {name}: {e}")
                
                # If we successfully processed the player, break the retry loop
                break
                
            except (TimeoutException, StaleElementReferenceException) as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed for {name}: {str(e)}")
                if retry_count == max_retries:
                    print(f"Failed to process {name} after {max_retries} attempts")
                time.sleep(10)  # Wait longer between retries

    # Instead of printing, save to JSON file
    # Create timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pga_stats_{timestamp}.json"
    
    # Write the data to a JSON file with nice formatting
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(players_data, f, indent=4)
    
    print(f"Data has been saved to {filename}")

finally:
    driver.quit()
