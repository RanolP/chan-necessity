// Imported first by main.ts, so loggers are configured before any feature
// module runs its install code.
import { configureLogging, KEY_LOG_LEVEL, toLogLevel } from "./shared/logtape.ts";

configureLogging(toLogLevel(GM_getValue(KEY_LOG_LEVEL)));
