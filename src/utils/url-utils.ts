import I18nKey from "@i18n/i18nKey";
import { i18n } from "@i18n/translation";

export function pathsEqual(path1: string, path2: string) {
	const normalizedPath1 = path1.replace(/^\/|\/$/g, "").toLowerCase();
	const normalizedPath2 = path2.replace(/^\/|\/$/g, "").toLowerCase();
	return normalizedPath1 === normalizedPath2;
}

function joinUrl(...parts: string[]): string {
	const joined = parts.join("/");
	return joined.replace(/\/+/g, "/");
}

export function getPostUrlBySlug(slug: string): string {
	return url(`/posts/${slug}/`);
}

export function getTagUrl(tag: string): string {
	if (!tag) return url("/archive/");
	return url(`/archive/?tag=${encodeURIComponent(tag.trim())}`);
}

function getParentCategoryParam(
	parentCategory: string | null | undefined,
): string {
	if (parentCategory === undefined) return "";
	return `parent=${encodeURIComponent(parentCategory?.trim() ?? "")}&`;
}

export function getParentCategoryFromSlug(slug: string): string {
	const segments = slug.split("/");
	if (segments.length < 2) return "";

	const parentCategory = segments[0].trim();
	try {
		return decodeURIComponent(parentCategory);
	} catch {
		return parentCategory;
	}
}

export function getParentCategoryUrl(parentCategory: string | null): string {
	return url(
		`/archive/?parent=${encodeURIComponent(parentCategory?.trim() ?? "")}`,
	);
}

export function getCategoryUrl(
	category: string | null,
	parentCategory?: string | null,
): string {
	const parentParam = getParentCategoryParam(parentCategory);
	if (
		!category ||
		category.trim() === "" ||
		category.trim().toLowerCase() === i18n(I18nKey.uncategorized).toLowerCase()
	)
		return url(`/archive/?${parentParam}uncategorized=true`);
	return url(
		`/archive/?${parentParam}category=${encodeURIComponent(category.trim())}`,
	);
}

export function getSubcategoryUrl(
	category: string | null,
	subcategory: string | null,
	parentCategory?: string | null,
): string {
	if (!subcategory || subcategory.trim() === "")
		return getCategoryUrl(category, parentCategory);
	if (!category || category.trim() === "") return url("/archive/");
	const parentParam = getParentCategoryParam(parentCategory);
	return url(
		`/archive/?${parentParam}category=${encodeURIComponent(category.trim())}&subcategory=${encodeURIComponent(subcategory.trim())}`,
	);
}

export function getDir(path: string): string {
	const lastSlashIndex = path.lastIndexOf("/");
	if (lastSlashIndex < 0) {
		return "/";
	}
	return path.substring(0, lastSlashIndex + 1);
}

export function url(path: string) {
	return joinUrl("", import.meta.env.BASE_URL, path);
}
