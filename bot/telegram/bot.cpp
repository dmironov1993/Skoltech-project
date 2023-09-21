#include "bot.h"

Bot::Bot(const std::string& api_endpoint) : bot_(std::make_unique<Client>(api_endpoint)) {

    int64_t update_id = 0;
    std::ifstream last_update_id_file;
    last_update_id_file.open(
        "/home/denis/Desktop/denis.mironov7/bot/telegram/bot_last_update_id.txt");
    // last_update_id_file.open("bot_last_update_id.txt");
    if (last_update_id_file.peek() != EOF) {
        last_update_id_file >> update_id;
    }
    last_update_id_file.close();
    update_id_ = update_id;

    bot_logfile_.open("/home/denis/Desktop/denis.mironov7/bot/telegram/bot_logfile.txt");
    bot_errorfile_.open("/home/denis/Desktop/denis.mironov7/bot/telegram/bot_errorfile.txt");

    // bot_logfile_.open("bot_logfile.txt");
    // bot_errorfile_.open("bot_errorfile.txt");
}

void Bot::Run() {
    bot_logfile_ << "BOT_CPP: Line = " + std::to_string(__LINE__) + ", Bot runs" << '\n';
    bot_errorfile_ << "BOT_CPP: Line = " + std::to_string(__LINE__) + ", Bot runs" << '\n';
    while (true) {
        std::vector<Message> messages = bot_->GetUpdate(5, update_id_);
        for (const Message& message : messages) {
            update_id_ = *message.GetUpdateId() + 1;
            std::string bot_message = bot_->HandleMessage(message);
            if (bot_message == "/random") {
                bot_logfile_ << "BOT_CPP: Line = " + std::to_string(__LINE__) +
                                    ", Bot has handled /random command"
                             << '\n';
                continue;
            } else if (bot_message == "/weather") {
                bot_logfile_ << "BOT_CPP: Line = " + std::to_string(__LINE__) +
                                    ", Bot has handled /weather command"
                             << '\n';
                continue;
            } else if (bot_message == "/styleguide") {
                bot_logfile_ << "BOT_CPP: Line = " + std::to_string(__LINE__) +
                                    ", Bot has handled /styleguide command"
                             << '\n';
                continue;
            } else if (bot_message == "/sticker") {
                bot_logfile_ << "BOT_CPP: Line = " + std::to_string(__LINE__) +
                                    ", Bot has handled /sticker command"
                             << '\n';
                continue;
            } else if (bot_message == "/gif") {
                bot_logfile_ << "BOT_CPP: Line = " + std::to_string(__LINE__) +
                                    ", Bot has handled /gif command "
                             << '\n';
                continue;
            } else if (bot_message == "/stop") {
                bot_logfile_ << "BOT_CPP: Line = " + std::to_string(__LINE__) +
                                    ", Bot has handled /stop command"
                             << '\n';
                bot_logfile_.close();
                bot_errorfile_.close();
                bot_.reset();
                return;
            } else if (bot_message == "/crash") {
                bot_logfile_.close();
                bot_errorfile_ << "BOT_CPP: Line = " + std::to_string(__LINE__) +
                                      ", /crash command has been executed"
                               << '\n';
                bot_errorfile_.close();
                bot_.reset();
                abort();
            }
        }
    }
}
