#include "bot.h"

int main() {
    const std::string api_endpoint =
        "https://api.telegram.org/bot5618873476:AAESVX1rOxfL6aPEuDTvZdLe4XT5u79kBAE";
    Bot bot(api_endpoint);
    bot.Run();

    std::cout << "BOT HAS STOPPED" << std::endl;
    return 0;
}
