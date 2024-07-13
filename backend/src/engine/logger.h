#ifndef ENGINE_LOGGER_H
#define ENGINE_LOGGER_H

#include <iostream>
#include <typeinfo>
#include <string>
#include <sstream>

enum class LogLevel {
    NONE = 0,
    DEBUG,
    INFO,
    WARN,
    ERROR
};

class Logger {
private:
    LogLevel logLevel;
    Logger() {
        logLevel = LogLevel::INFO;  // default log level
    }

public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLogLevel(LogLevel level) {
        logLevel = level;
    }

    template<typename T>
    void log(LogLevel level, const T& object, const std::string& message) {
        if(level >= logLevel) {
            std::stringstream s;
            s << "[" << getLevelName(level) << "] " 
              << typeid(object).name() << ": " 
              << message;
            std::cerr << s.str() << std::endl;
        }
    }

    std::string getLevelName(LogLevel level) {
        switch(level) {
            case LogLevel::DEBUG:
                return "DEBUG";
            case LogLevel::INFO:
                return "INFO";
            case LogLevel::WARN:
                return "WARNING";
            case LogLevel::ERROR:
                return "ERROR";
            default:
                return "UNKNOWN";
        }
    }
};

#define LOG_DEBUG(object, message) Logger::getInstance().log(LogLevel::DEBUG, object, message)
#define LOG_INFO(object, message) Logger::getInstance().log(LogLevel::INFO, object, message)
#define LOG_WARN(object, message) Logger::getInstance().log(LogLevel::WARN, object, message)
#define LOG_ERROR(object, message) Logger::getInstance().log(LogLevel::ERROR, object, message)


#endif