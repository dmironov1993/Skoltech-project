#include <fake/fake.h>
#include <telegram/bot.h>
#include <iostream>
#include <catch.hpp>

TEST_CASE("Single getMe") {
    telegram::FakeServer fake{"Single getMe"};
    fake.Start();

    const std::string api_endpoint = fake.GetUrl() + "bot123";
    Client client(api_endpoint);

    try {
        client.GetMe();
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    fake.StopAndCheckExpectations();
}

TEST_CASE("getMe error handling") {
    telegram::FakeServer fake{"getMe error handling"};
    fake.Start();

    const std::string api_endpoint = fake.GetUrl() + "bot123";
    Client client(api_endpoint);

    try {
        client.GetMe();
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    try {
        client.GetMe();
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    fake.StopAndCheckExpectations();
}

TEST_CASE("Single getUpdates and send messages") {
    telegram::FakeServer fake{"Single getUpdates and send messages"};
    fake.Start();

    const std::string api_endpoint = fake.GetUrl() + "bot123";
    Client client(api_endpoint);

    try {
        auto updates = client.GetUpdate();

        Message first_message = {"Hi!", updates[0].GetChatId(), {}, {}, {}, {}};
        client.SendMessage(first_message);

        Message second_message = {
            "Reply", updates[1].GetChatId(), updates[1].GetMessageId(), {}, {}, {}};
        client.SendMessage(second_message);
        client.SendMessage(second_message);

    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    fake.StopAndCheckExpectations();
}

TEST_CASE("Handle getUpdates offset") {
    telegram::FakeServer fake{"Handle getUpdates offset"};
    fake.Start();

    const std::string api_endpoint = fake.GetUrl() + "bot123";
    Client client(api_endpoint);

    auto updates = client.GetUpdate(5);
    REQUIRE(updates.size() == 2u);

    Message first_message = updates[0];
    Message second_message = updates[1];

    int64_t max_update_id = std::max(*first_message.GetUpdateId(), *second_message.GetUpdateId());

    try {
        updates = client.GetUpdate(5, max_update_id + 1);
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        updates.clear();
    }

    REQUIRE(updates.empty());

    try {
        updates = client.GetUpdate(5, max_update_id + 1);
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    REQUIRE(updates.size() == 1u);

    fake.StopAndCheckExpectations();
}
