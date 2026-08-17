import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const postsRoot = fileURLToPath(new URL("../src/content/posts/", import.meta.url));
const shouldWrite = process.argv.includes("--write");
const imageExtensions = new Set([
	".avif",
	".gif",
	".jpeg",
	".jpg",
	".png",
	".svg",
	".webp",
]);

async function walk(directory) {
	const entries = await readdir(directory, { withFileTypes: true });
	const files = [];
	for (const entry of entries) {
		const entryPath = path.join(directory, entry.name);
		if (entry.isDirectory()) files.push(...(await walk(entryPath)));
		else files.push(entryPath);
	}
	return files;
}

function normalize(value) {
	return value.replaceAll("\\", "/").toLocaleLowerCase("en-US");
}

function escapeAlt(value) {
	return value.replaceAll("\\", "\\\\").replaceAll("[", "\\[").replaceAll("]", "\\]");
}

function markdownDestination(relativePath) {
	const normalized = relativePath.replaceAll("\\", "/");
	return /[\s()]/u.test(normalized) ? `<${normalized}>` : normalized;
}

const allFiles = await walk(postsRoot);
const markdownFiles = allFiles.filter((file) => path.extname(file).toLowerCase() === ".md");
const imageFiles = allFiles.filter((file) => imageExtensions.has(path.extname(file).toLowerCase()));
const imageByNormalizedPath = new Map(imageFiles.map((file) => [normalize(file), file]));

let converted = 0;
let changedFiles = 0;
const unresolved = [];

for (const markdownFile of markdownFiles) {
	const articleDirectory = path.dirname(markdownFile);
	const source = await readFile(markdownFile, "utf8");
	let fileConversions = 0;

	const output = source.replace(/!\[\[([^\]\r\n]+)\]\]/gu, (original, rawReference) => {
		const [rawTarget, ...aliasParts] = rawReference.split("|");
		const target = rawTarget.trim().replace(/^\.\//u, "");
		if (!imageExtensions.has(path.extname(target).toLowerCase())) return original;

		const directCandidate = path.resolve(articleDirectory, target);
		let resolved = imageByNormalizedPath.get(normalize(directCandidate));

		if (!resolved) {
			const normalizedTarget = normalize(target);
			const localCandidates = imageFiles.filter((imageFile) => {
				const relativeToArticle = path.relative(articleDirectory, imageFile);
				return (
					!relativeToArticle.startsWith(`..${path.sep}`) &&
					(normalize(relativeToArticle) === normalizedTarget ||
						normalize(relativeToArticle).endsWith(`/${normalizedTarget}`) ||
						path.basename(imageFile).toLocaleLowerCase("en-US") ===
							path.basename(target).toLocaleLowerCase("en-US"))
				);
			});
			if (localCandidates.length === 1) resolved = localCandidates[0];
		}

		if (!resolved) {
			unresolved.push(`${path.relative(postsRoot, markdownFile)} -> ${target}`);
			return original;
		}

		const relativeImagePath = path.relative(articleDirectory, resolved);
		const alias = aliasParts.join("|").trim();
		const isDimension = /^\d+(?:x\d+)?$/u.test(alias);
		const alt = alias && !isDimension ? alias : path.parse(resolved).name.replaceAll("_", " ");
		fileConversions++;
		converted++;
		return `![${escapeAlt(alt)}](${markdownDestination(relativeImagePath)})`;
	});

	if (fileConversions > 0) {
		changedFiles++;
		if (shouldWrite) await writeFile(markdownFile, output, "utf8");
	}
}

console.log(`${shouldWrite ? "Converted" : "Would convert"} ${converted} image references in ${changedFiles} files.`);
if (unresolved.length > 0) {
	console.error(`Unresolved image references (${unresolved.length}):`);
	for (const item of unresolved) console.error(`- ${item}`);
	process.exitCode = 1;
}
