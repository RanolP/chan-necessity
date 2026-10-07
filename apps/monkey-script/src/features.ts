// Feature modules run in this order, as the sections of the original
// single-file script did. Each one installs itself on import.
import "./logging.ts";
import "./clip-for-firefox/firefox-ua.ts";
import "./shared/split-context.ts";
import "./deny-grid/index.ts";
import "./auto-claim-logs/index.ts";
import "./bookmarks/index.ts";
import "./shared/audio.ts";
import "./sound-panning/index.ts";
import "./captions/index.ts";
import "./split-view/index.ts";
import "./split-view/ambient-light.ts";
