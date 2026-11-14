# Copyright (c) Microsoft. All rights reserved.

"""Tests for conversation metadata validation in the DevUI conversation store and routes."""

import pytest
from fastapi.testclient import TestClient

from agent_framework_devui import DevServer
from agent_framework_devui._conversations import (
    ConversationMetadataValidationError,
    InMemoryConversationStore,
)


def test_create_conversation_rejects_non_map_metadata():
    """Metadata must be a map of string keys to string values."""
    store = InMemoryConversationStore()

    with pytest.raises(ConversationMetadataValidationError):
        store.create_conversation(metadata="oops-not-a-dict")


def test_create_conversation_rejects_non_string_metadata_values():
    """A metadata value that is not a string is rejected."""
    store = InMemoryConversationStore()

    with pytest.raises(ConversationMetadataValidationError):
        store.create_conversation(metadata={"agent_id": 7})


def test_create_conversation_accepts_none_metadata():
    """An explicit ``None`` is normalized to an empty map."""
    store = InMemoryConversationStore()

    conversation = store.create_conversation(metadata=None)

    assert conversation.metadata == {}


def test_update_conversation_rejects_invalid_metadata_and_keeps_previous_value():
    """A rejected update leaves the stored metadata untouched."""
    store = InMemoryConversationStore()
    created = store.create_conversation(metadata={"agent_id": "test_agent"})

    with pytest.raises(ConversationMetadataValidationError):
        store.update_conversation(created.id, metadata=["not", "a", "map"])

    assert store.get_conversation(created.id).metadata == {"agent_id": "test_agent"}


def test_update_conversation_normalizes_none_metadata():
    """An explicit ``None`` clears the metadata instead of storing it."""
    store = InMemoryConversationStore()
    created = store.create_conversation(metadata={"agent_id": "test_agent"})

    updated = store.update_conversation(created.id, metadata=None)

    assert updated.metadata == {}


@pytest.mark.asyncio
async def test_list_conversations_survives_legacy_invalid_metadata():
    """Metadata stored before validation existed must not break the list endpoint."""
    store = InMemoryConversationStore()
    created = store.create_conversation(metadata={"agent_id": "test_agent"})

    # Simulate a conversation written before the metadata was validated.
    store._conversations[created.id]["metadata"] = None

    listed = await store.list_conversations_by_metadata({})

    assert [conv.id for conv in listed] == [created.id]
    assert listed[0].metadata == {}


def test_create_conversation_route_rejects_invalid_metadata():
    """A create request with malformed metadata is a client error."""
    server = DevServer(mode="developer", auth_enabled=False)
    app = server.get_app()

    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/v1/conversations", json={"metadata": "oops-not-a-dict"})

        assert response.status_code == 400

        listing = client.get("/v1/conversations")
        assert listing.status_code == 200


def test_update_conversation_route_rejects_invalid_metadata():
    """An update request with malformed metadata is a client error, not a 404."""
    server = DevServer(mode="developer", auth_enabled=False)
    app = server.get_app()

    with TestClient(app, base_url="http://127.0.0.1") as client:
        created = client.post("/v1/conversations", json={"metadata": {"agent_id": "test_agent"}})
        assert created.status_code == 200
        conversation_id = created.json()["id"]

        response = client.post(f"/v1/conversations/{conversation_id}", json={"metadata": None})
        assert response.status_code == 200
        assert response.json()["metadata"] == {}

        invalid = client.post(f"/v1/conversations/{conversation_id}", json={"metadata": 5})
        assert invalid.status_code == 400

        listing = client.get("/v1/conversations")
        assert listing.status_code == 200
        assert [conv["id"] for conv in listing.json()["data"]] == [conversation_id]


def test_update_conversation_route_keeps_404_for_missing_conversation():
    """A missing conversation still answers with 404."""
    server = DevServer(mode="developer", auth_enabled=False)
    app = server.get_app()

    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/v1/conversations/does-not-exist", json={"metadata": {"agent_id": "a"}})

        assert response.status_code == 404
