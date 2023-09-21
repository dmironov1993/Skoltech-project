#include "client.h"

Message::Message(std::string text, int64_t chat_id, std::optional<int64_t> message_id,
                 std::optional<int64_t> update_id, std::optional<std::string> sticker_id,
                 std::optional<std::string> animation_id) {

    text_ = text;
    chat_id_ = chat_id;
    if (message_id) {
        message_id_ = message_id;
    }
    if (update_id) {
        update_id_ = update_id;
    }
    if (sticker_id) {
        sticker_id_ = sticker_id;
    }
    if (animation_id) {
        animation_id_ = animation_id;
    }
}

std::string Message::GetText() const {
    return text_;
}

int64_t Message::GetChatId() const {
    return chat_id_;
}

std::optional<int64_t> Message::GetMessageId() const {
    return message_id_;
}

std::optional<int64_t> Message::GetUpdateId() const {
    return update_id_;
}

std::optional<std::string> Message::GetStickerId() const {
    return sticker_id_;
}

std::optional<std::string> Message::GetAnimationId() const {
    return animation_id_;
}

Client::Client(const std::string& api_token) : api_token_(api_token), uri_(api_token) {

    client_logfile_.open("/home/denis/Desktop/denis.mironov7/bot/telegram/client_logfile.txt");
    client_errorfile_.open("/home/denis/Desktop/denis.mironov7/bot/telegram/client_errorfile.txt");

    // client_logfile_.open("client_logfile.txt");
    // client_errorfile_.open("client_errorfile.txt");

    if (api_token_.substr(0, 5) == "https") {
        session_.reset(new Poco::Net::HTTPSClientSession());
        client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                               ", HTTPS client session has been established"
                        << '\n';
    } else if (api_token_.substr(0, 4) == "http") {
        session_.reset(new Poco::Net::HTTPClientSession());
        client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                               ", HTTP client session has been established"
                        << '\n';
    } else {
        client_errorfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                                 ", MISSING HTTPS OR HTTP in api_token"
                          << '\n';
        client_errorfile_ << "CLIENT_CPP: LINE = " + std::to_string(__LINE__) +
                                 " no session has been established"
                          << '\n';
        throw std::invalid_argument("CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                                    " no session is established");
    }

    session_->setHost(uri_.getHost());
    client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                           ", client session set Host"
                    << '\n';

    session_->setPort(uri_.getPort());
    client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                           ", client session set Port"
                    << '\n';
}

void Client::GetMe() {
    Poco::URI uri(api_token_ + "/getMe");
    Poco::Net::HTTPRequest request("GET", uri.getPathAndQuery());
    session_->sendRequest(request);
    Poco::Net::HTTPResponse response;
    auto& stream = session_->receiveResponse(response);
    const auto status = response.getStatus();
    if (status != 200) {
        client_errorfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                                 " session status is " + std::to_string(status)
                          << '\n';
        throw std::invalid_argument("CLIENT_CPP: Line " + std::to_string(__LINE__) +
                                    " HTTPResponse-status is " + std::to_string(status) +
                                    " Expected 200");
    }
    (void)stream;
}

std::vector<Message> Client::GetUpdate(std::optional<int64_t> timeout,
                                       std::optional<int64_t> offset) {
    client_errorfile_ << '\n';
    Poco::URI uri(api_token_ + "/getUpdates");
    if (timeout) {
        uri.addQueryParameter("timeout", std::to_string(*timeout));
    }
    if (offset) {
        uri.addQueryParameter("offset", std::to_string(*offset));
    }
    Poco::Net::HTTPRequest request("GET", uri.getPathAndQuery());
    session_->sendRequest(request);
    Poco::Net::HTTPResponse response;
    auto& stream = session_->receiveResponse(response);
    const auto status = response.getStatus();
    if (status != 200) {
        client_errorfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                                 " session status is " + std::to_string(status)
                          << '\n';
        throw std::invalid_argument("CLIENT_CPP: Line " + std::to_string(__LINE__) +
                                    " HTTPResponse is " + std::to_string(status) + " Expected 200");
    }
    Poco::JSON::Parser parser;
    const auto result = parser.parse(stream);
    const auto array = result.extract<Poco::JSON::Object::Ptr>()->getArray("result");
    std::vector<Message> message_array;
    for (const auto& message : *array) {

        std::string text;
        try {
            text = message.extract<Poco::JSON::Object::Ptr>()
                       ->getObject("message")
                       ->getValue<std::string>("text");
        } catch (...) {
            client_errorfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                                     " text is empty in Message class"
                              << '\n';
        }

        int64_t chat_id = 0;
        try {
            chat_id = message.extract<Poco::JSON::Object::Ptr>()
                          ->getObject("message")
                          ->getObject("chat")
                          ->getValue<int64_t>("id");
        } catch (...) {
            client_errorfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                                     " chat_id is default (0) in Message class"
                              << '\n';
        }

        std::optional<int64_t> message_id;
        try {
            message_id =
                message.extract<Poco::JSON::Object::Ptr>()->getObject("message")->getValue<int64_t>(
                    "message_id");
        } catch (...) {
            client_errorfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                                     " message_id is empty in Message class"
                              << '\n';
        }

        std::optional<int64_t> update_id;
        try {
            update_id = message.extract<Poco::JSON::Object::Ptr>()->getValue<int64_t>("update_id");
        } catch (...) {
            client_errorfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                                     " update_id is empty in Message class"
                              << '\n';
        }

        std::optional<std::string> sticker_id;
        try {
            sticker_id = message.extract<Poco::JSON::Object::Ptr>()
                             ->getObject("message")
                             ->getObject("sticker")
                             ->getValue<std::string>("file_id");
        } catch (...) {
            client_errorfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                                     " sticker_id is empty in Message class"
                              << '\n';
        }

        std::optional<std::string> animation_id;
        try {
            animation_id = message.extract<Poco::JSON::Object::Ptr>()
                               ->getObject("message")
                               ->getObject("animation")
                               ->getValue<std::string>("file_id");
        } catch (...) {
            client_errorfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                                     " animation_id is empty in Message class"
                              << '\n';
        }

        client_errorfile_ << '\n';
        message_array.push_back(
            Message{text, chat_id, message_id, update_id, sticker_id, animation_id});
    }
    return message_array;
}

void Client::SendMessage(const Message& message) {
    Poco::URI uri(uri_.toString() + "/sendMessage");
    Poco::Net::HTTPRequest request(Poco::Net::HTTPRequest::HTTP_POST, uri.getPathAndQuery());
    Poco::JSON::Object message_json;
    message_json.set("text", message.GetText());
    message_json.set("chat_id", message.GetChatId());
    if (message.GetMessageId()) {
        message_json.set("reply_to_message_id", *message.GetMessageId());
    }
    std::stringstream ss;
    message_json.stringify(ss);
    request.setKeepAlive(true);
    request.setContentLength(ss.str().size());
    request.setContentType("application/json");
    std::ostream& out = session_->sendRequest(request);
    message_json.stringify(out);
    Poco::Net::HTTPResponse response;
    auto& stream = session_->receiveResponse(response);
    const auto status = response.getStatus();
    if (status != 200) {
        throw std::invalid_argument("CLIENT_CPP: Line " + std::to_string(__LINE__) +
                                    " HTTPResponse-status is " + std::to_string(status) +
                                    " Expected 200");
    }
    (void)stream;
}

void Client::SendSticker(const Message& message) {
    Poco::URI uri(uri_.toString() + "/sendSticker");
    Poco::Net::HTTPRequest request(Poco::Net::HTTPRequest::HTTP_POST, uri.getPathAndQuery());
    Poco::JSON::Object message_json;
    message_json.set("chat_id", message.GetChatId());
    message_json.set("sticker", *message.GetStickerId());
    if (message.GetMessageId()) {
        message_json.set("reply_to_message_id", *message.GetMessageId());
    }
    std::stringstream ss;
    message_json.stringify(ss);
    request.setKeepAlive(true);
    request.setContentLength(ss.str().size());
    request.setContentType("application/json");
    std::ostream& out = session_->sendRequest(request);
    message_json.stringify(out);
    Poco::Net::HTTPResponse response;
    auto& stream = session_->receiveResponse(response);
    const auto status = response.getStatus();
    if (status != 200) {
        throw std::invalid_argument("CLIENT_CPP: Line " + std::to_string(__LINE__) +
                                    " HTTPResponse-status is " + std::to_string(status) +
                                    " Expected 200");
    }
    (void)stream;
}

void Client::SendAnimation(const Message& message) {
    Poco::URI uri(uri_.toString() + "/sendAnimation");
    Poco::Net::HTTPRequest request(Poco::Net::HTTPRequest::HTTP_POST, uri.getPathAndQuery());
    Poco::JSON::Object message_json;
    message_json.set("chat_id", message.GetChatId());
    message_json.set("animation", *message.GetAnimationId());
    if (message.GetMessageId()) {
        message_json.set("reply_to_message_id", *message.GetMessageId());
    }
    std::stringstream ss;
    message_json.stringify(ss);
    request.setKeepAlive(true);
    request.setContentLength(ss.str().size());
    request.setContentType("application/json");
    std::ostream& out = session_->sendRequest(request);
    message_json.stringify(out);
    Poco::Net::HTTPResponse response;
    auto& stream = session_->receiveResponse(response);
    const auto status = response.getStatus();
    if (status != 200) {
        throw std::invalid_argument("CLIENT_CPP: Line " + std::to_string(__LINE__) +
                                    " HTTPResponse-status is " + std::to_string(status) +
                                    " Expected 200");
    }
    (void)stream;
}

std::string Client::HandleMessage(const Message& message) {
    if (message.GetText() == "/random") {

        std::uniform_int_distribution<int> distribution(1, kMAX);
        Message new_message(std::to_string(distribution(generator)), message.GetChatId(), {}, {},
                            {}, {});

        client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                               " /random message has been sent"
                        << "\n";
        SendMessage(new_message);
        return "/random";

    } else if (message.GetText() == "/weather") {

        Message new_message("Winter Is Coming", message.GetChatId(), {}, {}, {}, {});

        client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                               " /weather message has been sent"
                        << "\n";
        SendMessage(new_message);
        return "/weather";

    } else if (message.GetText() == "/styleguide") {

        Message new_message("YOU SHALL NOT PASS THE CODE REVIEW HORAAAAAAAAAAAAAY",
                            message.GetChatId(), {}, {}, {}, {});

        client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                               " /styleguide message has been sent"
                        << "\n";
        SendMessage(new_message);
        return "/styleguide";

    } else if (message.GetText() == "/sticker") {

        Message new_message(
            "", message.GetChatId(), {}, {},
            "CAACAgUAAxkBAAETVydiZiwhH_fMn2x-CK8ZZSHklAREWQACHgAD9oK6D71hz3MWpjrmJAQ", {});

        client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                               " /sticker message has been sent"
                        << "\n";
        SendSticker(new_message);
        return "/sticker";

    } else if (message.GetText() == "/gif") {

        Message new_message("", message.GetChatId(), {}, {}, {},
                            "https://media.giphy.com/media/3o6fJgacEvNBZZs6IM/giphy.gif");

        client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                               " /gif message has been sent"
                        << "\n";
        SendAnimation(new_message);
        return "/gif";

    } else if (message.GetText() == "/stop") {

        client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                               " /stop message has been sent"
                        << "\n";
        std::ofstream last_update_id_file;
        last_update_id_file.open(
            "/home/denis/Desktop/denis.mironov7/bot/telegram/bot_last_update_id.txt");
        // last_update_id_file.open("bot_last_update_id.txt");
        last_update_id_file << *message.GetUpdateId() + 1;
        last_update_id_file.close();
        return "/stop";

    } else if (message.GetText() == "/crash") {

        client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                               " /crash message has been sent"
                        << "\n";
        std::ofstream last_update_id_file;
        last_update_id_file.open(
            "/home/denis/Desktop/denis.mironov7/bot/telegram/bot_last_update_id.txt");
        // last_update_id_file.open("bot_last_update_id.txt");
        last_update_id_file << *message.GetUpdateId() + 1;
        last_update_id_file.close();
        return "/crash";

    } else {

        Message new_message(
            "Your command is not understood. Please select a valid command "
            "from the following list: \n"
            "/random     :   displays some random number \n"
            "/weather    :   says, \"Winter Is Coming\" \n"
            "/styleguide :   a bot will make a joke \n"
            "/sticker    :   a bot sends some sticker \n"
            "/gif        :   a bot sends some gif \n"
            "/stop       :   a bot stops \n"
            "/abort      :   a bot stop with exit code other than 0",
            message.GetChatId(), {}, {}, {}, {});

        client_logfile_ << "CLIENT_CPP: Line = " + std::to_string(__LINE__) +
                               " unrecognized message = (" + new_message.GetText() +
                               ") has been sent"
                        << "\n";
        SendMessage(new_message);
    }
    return "";
}
