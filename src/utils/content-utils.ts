import { type CollectionEntry, getCollection } from "astro:content";
import I18nKey from "@i18n/i18nKey";
import { i18n } from "@i18n/translation";
import { getCategoryUrl } from "@utils/url-utils.ts";

// // Retrieve posts and sort them by publication date
async function getRawSortedPosts() {
	const allBlogPosts = await getCollection("posts", ({ data }) => {
		return import.meta.env.PROD ? data.draft !== true : true;
	});

	const sorted = allBlogPosts.sort((a, b) => {
		const dateA = new Date(a.data.published);
		const dateB = new Date(b.data.published);
		return dateA > dateB ? -1 : 1;
	});
	return sorted;
}

const naturalCollator = new Intl.Collator("zh-CN", {
	numeric: true,
	sensitivity: "base",
});

function sequenceSegmentKey(segment: string) {
	const decoded = decodeURIComponent(segment).replace(/\.md$/iu, "");
	const namedSequence = decoded.match(/^(?:c|chapter|kapitel)[\s_-]*(\d+)(.*)$/iu);
	const numberedSequence = decoded.match(/^(\d+)(?:\.(\d+))?(.*)$/u);
	const match = namedSequence ?? numberedSequence;
	if (!match) return { sequence: false, numbers: [] as number[], remainder: decoded };

	return {
		sequence: true,
		numbers: match
			.slice(1, -1)
			.filter((part): part is string => typeof part === "string")
			.map(Number),
		remainder: match.at(-1) ?? "",
	};
}

function comparePathNaturally(slugA: string, slugB: string) {
	const segmentsA = slugA.split("/");
	const segmentsB = slugB.split("/");
	const length = Math.max(segmentsA.length, segmentsB.length);

	for (let i = 0; i < length; i++) {
		if (segmentsA[i] === undefined) return -1;
		if (segmentsB[i] === undefined) return 1;

		const keyA = sequenceSegmentKey(segmentsA[i]);
		const keyB = sequenceSegmentKey(segmentsB[i]);
		if (keyA.sequence !== keyB.sequence) return keyA.sequence ? -1 : 1;

		for (let j = 0; j < Math.max(keyA.numbers.length, keyB.numbers.length); j++) {
			const numberA = keyA.numbers[j] ?? -1;
			const numberB = keyB.numbers[j] ?? -1;
			if (numberA !== numberB) return numberA - numberB;
		}

		const remainderComparison = naturalCollator.compare(keyA.remainder, keyB.remainder);
		if (remainderComparison !== 0) return remainderComparison;
	}
	return 0;
}

export async function getSortedPosts() {
	const sorted = await getRawSortedPosts();

	for (let i = 1; i < sorted.length; i++) {
		sorted[i].data.nextSlug = sorted[i - 1].slug;
		sorted[i].data.nextTitle = sorted[i - 1].data.title;
	}
	for (let i = 0; i < sorted.length - 1; i++) {
		sorted[i].data.prevSlug = sorted[i + 1].slug;
		sorted[i].data.prevTitle = sorted[i + 1].data.title;
	}

	return sorted;
}
export type PostForList = {
	slug: string;
	data: CollectionEntry<"posts">["data"];
};
export async function getSortedPostsList(): Promise<PostForList[]> {
	const sortedFullPosts = await getRawSortedPosts();
	sortedFullPosts.sort((a, b) => {
		const yearDifference = b.data.published.getFullYear() - a.data.published.getFullYear();
		return yearDifference || comparePathNaturally(a.slug, b.slug);
	});

	// delete post.body
	const sortedPostsList = sortedFullPosts.map((post) => ({
		slug: post.slug,
		data: post.data,
	}));

	return sortedPostsList;
}
export type Tag = {
	name: string;
	count: number;
};

export async function getTagList(): Promise<Tag[]> {
	const allBlogPosts = await getCollection<"posts">("posts", ({ data }) => {
		return import.meta.env.PROD ? data.draft !== true : true;
	});

	const countMap: { [key: string]: number } = {};
	allBlogPosts.forEach((post: { data: { tags: string[] } }) => {
		post.data.tags.forEach((tag: string) => {
			if (!countMap[tag]) countMap[tag] = 0;
			countMap[tag]++;
		});
	});

	// sort tags
	const keys: string[] = Object.keys(countMap).sort((a, b) => {
		return a.toLowerCase().localeCompare(b.toLowerCase());
	});

	return keys.map((key) => ({ name: key, count: countMap[key] }));
}

export type Category = {
	name: string;
	count: number;
	url: string;
};

export async function getCategoryList(): Promise<Category[]> {
	const allBlogPosts = await getCollection<"posts">("posts", ({ data }) => {
		return import.meta.env.PROD ? data.draft !== true : true;
	});
	const count: { [key: string]: number } = {};
	allBlogPosts.forEach((post: { data: { category: string | null } }) => {
		if (!post.data.category) {
			const ucKey = i18n(I18nKey.uncategorized);
			count[ucKey] = count[ucKey] ? count[ucKey] + 1 : 1;
			return;
		}

		const categoryName =
			typeof post.data.category === "string"
				? post.data.category.trim()
				: String(post.data.category).trim();

		count[categoryName] = count[categoryName] ? count[categoryName] + 1 : 1;
	});

	const lst = Object.keys(count).sort((a, b) => {
		return a.toLowerCase().localeCompare(b.toLowerCase());
	});

	const ret: Category[] = [];
	for (const c of lst) {
		ret.push({
			name: c,
			count: count[c],
			url: getCategoryUrl(c),
		});
	}
	return ret;
}
