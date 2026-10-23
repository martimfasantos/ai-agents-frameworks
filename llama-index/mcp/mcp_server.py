import re
import httpx
import argparse
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

# 1. Create MCP server
mcp = MCPServer(name="Example MCP Server")

# 2. Create the output models for the tool
class DriverInfo(BaseModel):
    id: str
    full_name: str
    name_acronym: str
    team_name: str

# 3. Create the tool and register them with the MCP server
@mcp.tool()
async def get_driver_info(driver_number: int) -> DriverInfo:
    """Get F1 driver info by driver ID

    Args:
        driver_id(str): The ID of the driver to get the info for.
    Returns:
        DriverInfo: The info of the specified driver.
    """
    url = f"https://api.openf1.org/v1/drivers?driver_number={driver_number}&session_key=latest"
    response = httpx.get(url) # External API call example

    if response.status_code == 200:
        data = response.json()
        if data and len(data) > 0:
            driver_data = data[0]
            return DriverInfo(
                id=str(driver_data["driver_number"]),
                full_name=driver_data["full_name"],
                name_acronym=driver_data["name_acronym"],
                team_name=driver_data["team_name"]
            )
    
    return DriverInfo(
        id=str(driver_number),
        full_name="Unknown Driver",
        name_acronym="UNK",
        team_name="Unknown Team"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type", type=str, default="sse", choices=["sse", "stdio"]
    )
    args = parser.parse_args()
    mcp.run(args.server_type)