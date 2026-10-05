// Logging for every feature, page side and captions worker alike. Each
// feature logs under ["chan-necessity", <its directory name>, ...]; the
// worker runs in its own global, so it calls configureLogging itself.
import { configureSync, getConsoleSink, getLogger as getRootLogger, isLogLevel, type Logger, type LogLevel } from "@logtape/logtape";

export const ROOT_CATEGORY = "chan-necessity";
// Set to "debug" (any LogTape level works) in the Violentmonkey script's
// Values tab to see debug logs; the default is "info".
export const KEY_LOG_LEVEL = "dev.logLevel";
export const DEFAULT_LOG_LEVEL: LogLevel = "info";

export const toLogLevel = (value: unknown): LogLevel => (typeof value === "string" && isLogLevel(value) ? value : DEFAULT_LOG_LEVEL);

export function configureLogging(level: LogLevel): void {
    configureSync({
        reset: true,
        sinks: { console: getConsoleSink() },
        loggers: [
            { category: ROOT_CATEGORY, sinks: ["console"], lowestLevel: level },
            { category: ["logtape", "meta"], sinks: ["console"], lowestLevel: "warning" },
        ],
    });
}

export const getLogger = (category: readonly string[]): Logger => getRootLogger([ROOT_CATEGORY, ...category]);
