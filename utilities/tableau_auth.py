"""
Tableau Authentication Module
Handles authentication using Tableau REST API /signin endpoint
"""

import os
import httpx
import xml.etree.ElementTree as ET
from typing import Optional, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Tableau Server Configuration
TABLEAU_SERVER = os.getenv("TABLEAU_DOMAIN_FULL", "https://prod-ca-a.online.tableau.com")
TABLEAU_SITE = os.getenv("TABLEAU_SITE", "")
TABLEAU_API_VERSION = os.getenv("TABLEAU_API_VERSION", "3.23")

# Admin roles that are allowed to use the admin portal
ALLOWED_ROLES = {
    "SiteAdministratorCreator",
    "SiteAdministratorExplorer"
}


class TableauAuthToken:
    """Represents a Tableau authentication token with user info"""

    def __init__(self, token: str, site_id: str, user_id: str, user_name: str,
                 site_role: str, expires_at: datetime):
        self.token = token
        self.site_id = site_id
        self.user_id = user_id
        self.user_name = user_name
        self.site_role = site_role
        self.expires_at = expires_at

    def is_expired(self) -> bool:
        """Check if the token has expired"""
        return datetime.now() >= self.expires_at

    def is_admin(self) -> bool:
        """Check if the user has admin privileges"""
        return self.site_role in ALLOWED_ROLES

    def to_dict(self) -> Dict:
        """Convert to dictionary for session storage"""
        return {
            "token": self.token,
            "site_id": self.site_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "site_role": self.site_role,
            "expires_at": self.expires_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TableauAuthToken':
        """Create from dictionary (from session)"""
        return cls(
            token=data["token"],
            site_id=data["site_id"],
            user_id=data["user_id"],
            user_name=data["user_name"],
            site_role=data["site_role"],
            expires_at=datetime.fromisoformat(data["expires_at"])
        )


async def authenticate_with_tableau(username: str, password: str) -> TableauAuthToken:
    """
    Authenticate with Tableau REST API using username/password

    Args:
        username: Tableau username (email)
        password: Tableau password

    Returns:
        TableauAuthToken with user info and auth token

    Raises:
        Exception: If authentication fails
    """

    # Build the signin request XML
    signin_xml = f"""
    <tsRequest>
        <credentials name="{username}" password="{password}">
            <site contentUrl="{TABLEAU_SITE}" />
        </credentials>
    </tsRequest>
    """

    url = f"{TABLEAU_SERVER}/api/{TABLEAU_API_VERSION}/auth/signin"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                content=signin_xml.strip(),
                headers={"Content-Type": "application/xml"},
                timeout=30.0
            )

            if response.status_code != 200:
                error_detail = _parse_error_from_xml(response.text)
                raise Exception(f"Authentication failed: {error_detail}")

            # Parse the XML response
            root = ET.fromstring(response.text)

            # Debug: print the full response
            print(f"🔍 Tableau signin response:\n{response.text}")

            # Extract credentials element
            credentials = root.find(".//{http://tableau.com/api}credentials")
            if credentials is None:
                raise Exception("Invalid response format from Tableau")

            token = credentials.get("token")
            site = credentials.find(".//{http://tableau.com/api}site")
            user = credentials.find(".//{http://tableau.com/api}user")

            if not token or site is None or user is None:
                raise Exception("Missing authentication data in response")

            site_id = site.get("id")
            user_id = user.get("id")

            # Try to get siteRole from user element
            site_role = user.get("siteRole")

            # Debug: print what we extracted
            print(f"🔍 Extracted - user_id: {user_id}, site_role: {site_role}")
            print(f"🔍 User element attributes: {user.attrib}")

            # If siteRole is not in the signin response, we need to fetch user details
            if not site_role:
                print(f"⚠️ siteRole not in signin response, fetching user details...")
                user_url = f"{TABLEAU_SERVER}/api/{TABLEAU_API_VERSION}/sites/{site_id}/users/{user_id}"
                user_response = await client.get(
                    user_url,
                    headers={"X-Tableau-Auth": token},
                    timeout=10.0
                )

                if user_response.status_code == 200:
                    user_root = ET.fromstring(user_response.text)
                    user_detail = user_root.find(".//{http://tableau.com/api}user")
                    if user_detail is not None:
                        site_role = user_detail.get("siteRole")
                        print(f"✅ Fetched siteRole from user details: {site_role}")

            if not site_role:
                raise Exception("Could not determine user's site role")

            # Token typically expires in 4 hours
            expires_at = datetime.now() + timedelta(hours=4)

            return TableauAuthToken(
                token=token,
                site_id=site_id,
                user_id=user_id,
                user_name=username,
                site_role=site_role,
                expires_at=expires_at
            )

        except httpx.RequestError as e:
            raise Exception(f"Connection error: {str(e)}")


async def authenticate_with_pat(token_name: str, token_value: str) -> TableauAuthToken:
    """
    Authenticate with Tableau REST API using Personal Access Token (PAT)

    Args:
        token_name: PAT name
        token_value: PAT secret value

    Returns:
        TableauAuthToken with user info and auth token

    Raises:
        Exception: If authentication fails
    """

    # Build the signin request XML for PAT
    signin_xml = f"""
    <tsRequest>
        <credentials personalAccessTokenName="{token_name}"
                     personalAccessTokenSecret="{token_value}">
            <site contentUrl="{TABLEAU_SITE}" />
        </credentials>
    </tsRequest>
    """

    url = f"{TABLEAU_SERVER}/api/{TABLEAU_API_VERSION}/auth/signin"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                content=signin_xml.strip(),
                headers={"Content-Type": "application/xml"},
                timeout=30.0
            )

            if response.status_code != 200:
                error_detail = _parse_error_from_xml(response.text)
                raise Exception(f"PAT authentication failed: {error_detail}")

            # Parse the XML response (same as password auth)
            root = ET.fromstring(response.text)

            print(f"🔍 PAT signin response:\n{response.text}")

            credentials = root.find(".//{http://tableau.com/api}credentials")

            if credentials is None:
                raise Exception("Invalid response format from Tableau")

            token = credentials.get("token")
            site = credentials.find(".//{http://tableau.com/api}site")
            user = credentials.find(".//{http://tableau.com/api}user")

            if not token or site is None or user is None:
                raise Exception("Missing authentication data in response")

            site_id = site.get("id")
            user_id = user.get("id")
            site_role = user.get("siteRole")

            print(f"🔍 PAT - Extracted user_id: {user_id}, site_role: {site_role}")

            # If siteRole is not in the signin response, fetch user details
            if not site_role:
                print(f"⚠️ PAT - siteRole not in signin response, fetching user details...")
                user_url = f"{TABLEAU_SERVER}/api/{TABLEAU_API_VERSION}/sites/{site_id}/users/{user_id}"
                user_response = await client.get(
                    user_url,
                    headers={"X-Tableau-Auth": token},
                    timeout=10.0
                )

                if user_response.status_code == 200:
                    user_root = ET.fromstring(user_response.text)
                    user_detail = user_root.find(".//{http://tableau.com/api}user")
                    if user_detail is not None:
                        site_role = user_detail.get("siteRole")
                        print(f"✅ PAT - Fetched siteRole: {site_role}")

            if not site_role:
                raise Exception("Could not determine user's site role")

            expires_at = datetime.now() + timedelta(hours=4)

            return TableauAuthToken(
                token=token,
                site_id=site_id,
                user_id=user_id,
                user_name=token_name,  # Use token name as identifier
                site_role=site_role,
                expires_at=expires_at
            )

        except httpx.RequestError as e:
            raise Exception(f"Connection error: {str(e)}")


async def signout_from_tableau(auth_token: str) -> None:
    """
    Sign out from Tableau (invalidate the token)

    Args:
        auth_token: Tableau auth token to invalidate
    """
    url = f"{TABLEAU_SERVER}/api/{TABLEAU_API_VERSION}/auth/signout"

    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                url,
                headers={
                    "X-Tableau-Auth": auth_token
                },
                timeout=10.0
            )
        except Exception as e:
            # Signout failure is not critical
            print(f"Warning: Failed to sign out from Tableau: {e}")


def _parse_error_from_xml(xml_text: str) -> str:
    """Parse error message from Tableau XML response"""
    try:
        root = ET.fromstring(xml_text)
        error = root.find(".//{http://tableau.com/api}error")
        if error is not None:
            summary = error.get("summary", "Unknown error")
            detail = error.get("detail", "")
            return f"{summary} {detail}".strip()
    except:
        pass
    return "Authentication failed"


async def verify_admin_access(auth_token: TableauAuthToken) -> bool:
    """
    Verify that the authenticated user has admin access

    Args:
        auth_token: Tableau authentication token

    Returns:
        True if user is an admin, False otherwise
    """
    if auth_token.is_expired():
        return False

    return auth_token.is_admin()
