#pragma once

#include <string>
#include <fstream>
#include "client.h"

class Bot {
public:
    Bot(const std::string& api_endpoint);
    void Run();

private:
    std::unique_ptr<Client> bot_;
    int64_t update_id_;
    std::ofstream bot_logfile_;
    std::ofstream bot_errorfile_;
};
