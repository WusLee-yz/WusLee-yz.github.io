import { type CollectionEntry, getCollection } from "astro:content";
import I18nKey from "@i18n/i18nKey";
import { i18n } from "@i18n/translation";
import {
	getCategoryUrl,
	getParentCategoryFromSlug,
	getParentCategoryUrl,
	getSubcategoryUrl,
} from "@utils/url-utils.ts";

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

type Eco2011ArchiveKey = {
	seriesPath: string;
	unit: number | null;
};

function getEco2011ArchiveKey(slug: string): Eco2011ArchiveKey | null {
	const segments = slug.split("/");
	const seriesIndex = segments.findIndex(
		(segment) => decodeURIComponent(segment).toLowerCase() === "eco2011",
	);
	if (seriesIndex === -1) return null;

	const unitSegment = decodeURIComponent(segments[seriesIndex + 1] ?? "");
	const unitMatch = unitSegment.match(/^c[\s_-]*(\d+)(?:[\s_-]|$)/iu);

	return {
		seriesPath: segments.slice(0, seriesIndex + 1).join("/").toLowerCase(),
		unit: unitMatch ? Number(unitMatch[1]) : null,
	};
}

function compareEco2011ArchivePosts(
	a: CollectionEntry<"posts">,
	b: CollectionEntry<"posts">,
) {
	const keyA = getEco2011ArchiveKey(a.slug);
	const keyB = getEco2011ArchiveKey(b.slug);
	if (!keyA || !keyB || keyA.seriesPath !== keyB.seriesPath) return 0;

	if (keyA.unit !== null && keyB.unit !== null) {
		return keyA.unit - keyB.unit;
	}
	if (keyA.unit !== null) return -1;
	if (keyB.unit !== null) return 1;

	const dateDifference = b.data.published.getTime() - a.data.published.getTime();
	return dateDifference || comparePathNaturally(a.slug, b.slug);
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
		if (yearDifference) return yearDifference;

		const eco2011Difference = compareEco2011ArchivePosts(a, b);
		return eco2011Difference || comparePathNaturally(a.slug, b.slug);
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

export type Subcategory = {
	name: string;
	count: number;
	url: string;
};

export type Category = {
	name: string;
	count: number;
	url: string;
	children: Subcategory[];
};

export type ParentCategory = {
	value: string;
	name: string;
	count: number;
	url: string;
	children: Category[];
};

type CategoryCount = {
	count: number;
	subcategories: Record<string, number>;
};

type ParentCategoryCount = {
	count: number;
	categories: Record<string, CategoryCount>;
};

export async function getCategoryList(): Promise<ParentCategory[]> {
	const allBlogPosts = await getCollection<"posts">("posts", ({ data }) => {
		return import.meta.env.PROD ? data.draft !== true : true;
	});
	const count: Record<string, ParentCategoryCount> = {};
	const uncategorizedName = i18n(I18nKey.uncategorized);

	allBlogPosts.forEach((post) => {
		const parentCategory = getParentCategoryFromSlug(post.slug);
		const categoryName = post.data.category?.trim() || uncategorizedName;

		count[parentCategory] ??= { count: 0, categories: {} };
		count[parentCategory].count++;

		count[parentCategory].categories[categoryName] ??= {
			count: 0,
			subcategories: {},
		};
		const category = count[parentCategory].categories[categoryName];
		category.count++;

		const subcategoryName = post.data.subcategory?.trim();
		if (subcategoryName) {
			category.subcategories[subcategoryName] =
				(category.subcategories[subcategoryName] ?? 0) + 1;
		}
	});

	return Object.keys(count)
		.sort((a, b) =>
			naturalCollator.compare(a || uncategorizedName, b || uncategorizedName),
		)
		.map((parentCategory) => ({
			value: parentCategory,
			name: parentCategory || uncategorizedName,
			count: count[parentCategory].count,
			url: getParentCategoryUrl(parentCategory),
			children: Object.keys(count[parentCategory].categories)
				.sort((a, b) => naturalCollator.compare(a, b))
				.map((categoryName) => {
					const category = count[parentCategory].categories[categoryName];
					return {
						name: categoryName,
						count: category.count,
						url: getCategoryUrl(categoryName, parentCategory),
						children: Object.keys(category.subcategories)
							.sort((a, b) => naturalCollator.compare(a, b))
							.map((subcategory) => ({
								name: subcategory,
								count: category.subcategories[subcategory],
								url: getSubcategoryUrl(
									categoryName,
									subcategory,
									parentCategory,
								),
							})),
					};
				}),
		}));
}
