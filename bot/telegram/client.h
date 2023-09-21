#pragma once

#include <string>
#include <vector>
#include <optional>
#include <random>
#include <limits>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <Poco/URI.h>
#include <Poco/JSON/Object.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/JSON/Parser.h>

const int kMAX = std::numeric_limits<int>::max();

static std::random_device rd;
static std::mt19937 generator(rd());

class Message {
public:
    Message(std::string text, int64_t chat_id, std::optional<int64_t> message_id,
            std::optional<int64_t> update_id, std::optional<std::string> sticker_id,
            std::optional<std::string> animation_id);

    std::string GetText() const;
    int64_t GetChatId() const;
    std::optional<int64_t> GetMessageId() const;
    std::optional<int64_t> GetUpdateId() const;
    std::optional<std::string> GetStickerId() const;
    std::optional<std::string> GetAnimationId() const;

private:
    std::string text_;
    int64_t chat_id_;
    std::optional<int64_t> message_id_;
    std::optional<int64_t> update_id_;
    std::optional<std::string> sticker_id_;
    std::optional<std::string> animation_id_;
};

class Client {
public:
    Client(const std::string& api_token);
    void GetMe();
    std::vector<Message> GetUpdate(std::optional<int64_t> timeout = {},
                                   std::optional<int64_t> update_id = {});
    void SendMessage(const Message& message);
    void SendSticker(const Message& message);
    void SendAnimation(const Message& message);
    std::string HandleMessage(const Message& message);

private:
    const std::string api_token_;
    const Poco::URI uri_;
    std::unique_ptr<Poco::Net::HTTPClientSession> session_;
    std::ofstream client_logfile_;
    std::ofstream client_errorfile_;
};
