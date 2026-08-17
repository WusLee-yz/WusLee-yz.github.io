import { visit } from "unist-util-visit";

function escapeHtml(value) {
	return value
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;");
}

export function remarkMermaid() {
	return (tree) => {
		visit(tree, "code", (node, index, parent) => {
			if (node.lang?.toLowerCase() !== "mermaid" || index === undefined || !parent) {
				return;
			}

			parent.children[index] = {
				type: "html",
				value: `<div class="mermaid-wrapper" data-pagefind-ignore><pre class="mermaid">${escapeHtml(node.value)}</pre></div>`,
			};
		});
	};
}
