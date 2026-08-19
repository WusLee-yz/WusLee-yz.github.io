import { h } from "hastscript";

/**
 * Render a reusable music player from a leaf Markdown directive.
 *
 * Usage:
 * ::music{src="/audio.mp3" title="Song" artist="Artist" album="Album" cover="/cover.jpg"}
 */
export function MusicPlayerComponent(properties, children) {
	if (Array.isArray(children) && children.length !== 0) {
		return h(
			"div",
			{ class: "hidden" },
			'Invalid directive. ("music" directive must be a leaf directive.)',
		);
	}

	const { src, title, artist, album, cover } = properties;
	if (!src || !title || !artist) {
		return h(
			"div",
			{ class: "hidden" },
			'Invalid music directive. ("src", "title", and "artist" are required.)',
		);
	}

	const metadata = album ? `${artist} · ${album}` : artist;
	const body = h("div", { class: "music-player-body" }, [
		h("div", { class: "music-player-title" }, title),
		h("div", { class: "music-player-meta" }, metadata),
		h(
			"audio",
			{
				class: "music-player-controls",
				controls: true,
				controlsList: "nodownload",
				disableRemotePlayback: true,
				preload: "metadata",
				src,
				"aria-label": `${title} — ${artist}`,
			},
			"Your browser does not support audio playback.",
		),
	]);

	return h("figure", { class: "music-player" }, [
		cover
			? h("img", {
					class: "music-player-cover",
					src: cover,
					alt: `${title} cover`,
					loading: "lazy",
					decoding: "async",
				})
			: h(
					"div",
					{
						class: "music-player-cover music-player-cover-placeholder",
						"aria-hidden": "true",
					},
					"♪",
				),
		body,
	]);
}
