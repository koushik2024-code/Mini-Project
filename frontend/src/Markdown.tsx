import { type ReactNode } from 'react'

/**
 * Small markdown renderer for model output.
 *
 * Deliberately dependency-free and deliberately limited: it covers what the
 * routed models actually emit -- headings, bullet and numbered lists, fenced
 * and inline code, bold, italic, rules -- and nothing else. Tables fall
 * through to preformatted text rather than being half-rendered.
 *
 * Nothing here interprets raw HTML, so model output cannot inject markup.
 */

interface MarkdownProps {
  content: string
}

// Inline: **bold**, *italic*/_italic_, `code`. Split on the whole set at once
// so the pieces cannot nest incorrectly.
const INLINE = /(\*\*[^*]+\*\*|__[^_]+__|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`)/g

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  return text.split(INLINE).filter(Boolean).map((piece, i) => {
    const key = `${keyPrefix}-${i}`

    if (piece.startsWith('**') && piece.endsWith('**')) {
      return <strong key={key}>{piece.slice(2, -2)}</strong>
    }
    if (piece.startsWith('__') && piece.endsWith('__')) {
      return <strong key={key}>{piece.slice(2, -2)}</strong>
    }
    if (piece.startsWith('`') && piece.endsWith('`')) {
      return <code key={key}>{piece.slice(1, -1)}</code>
    }
    if (
      (piece.startsWith('*') && piece.endsWith('*')) ||
      (piece.startsWith('_') && piece.endsWith('_'))
    ) {
      return <em key={key}>{piece.slice(1, -1)}</em>
    }
    return <span key={key}>{piece}</span>
  })
}

function Markdown({ content }: MarkdownProps) {
  const blocks: ReactNode[] = []
  const lines = content.split('\n')

  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i]

    // Fenced code block
    if (line.trimStart().startsWith('```')) {
      const body: string[] = []
      i++
      while (i < lines.length && !lines[i].trimStart().startsWith('```')) {
        body.push(lines[i])
        i++
      }
      i++ // closing fence
      blocks.push(
        <pre key={key++}><code>{body.join('\n')}</code></pre>
      )
      continue
    }

    // Blank line
    if (!line.trim()) {
      i++
      continue
    }

    // Horizontal rule
    if (/^\s*([-*_])\s*(\1\s*){2,}$/.test(line)) {
      blocks.push(<hr key={key++} />)
      i++
      continue
    }

    // Heading
    const heading = /^(#{1,6})\s+(.*)$/.exec(line)
    if (heading) {
      const level = Math.min(heading[1].length, 3)
      const text = renderInline(heading[2], `h${key}`)
      if (level === 1) blocks.push(<h1 key={key++}>{text}</h1>)
      else if (level === 2) blocks.push(<h2 key={key++}>{text}</h2>)
      else blocks.push(<h3 key={key++}>{text}</h3>)
      i++
      continue
    }

    // Table -- not rendered as a table on purpose; kept legible as-is
    if (line.includes('|') && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
      const body: string[] = []
      while (i < lines.length && lines[i].includes('|')) {
        body.push(lines[i])
        i++
      }
      blocks.push(<pre key={key++}><code>{body.join('\n')}</code></pre>)
      continue
    }

    // Bulleted list
    if (/^\s*[-*+]\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*+]\s+/, ''))
        i++
      }
      blocks.push(
        <ul key={key++}>
          {items.map((item, n) => <li key={n}>{renderInline(item, `ul${key}-${n}`)}</li>)}
        </ul>
      )
      continue
    }

    // Numbered list
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+[.)]\s+/, ''))
        i++
      }
      blocks.push(
        <ol key={key++}>
          {items.map((item, n) => <li key={n}>{renderInline(item, `ol${key}-${n}`)}</li>)}
        </ol>
      )
      continue
    }

    // Paragraph -- consume until a blank line or a block starter
    const para: string[] = []
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i]) &&
      !/^#{1,6}\s+/.test(lines[i]) &&
      !lines[i].trimStart().startsWith('```')
    ) {
      para.push(lines[i])
      i++
    }
    blocks.push(<p key={key++}>{renderInline(para.join(' '), `p${key}`)}</p>)
  }

  return <div className="md">{blocks}</div>
}

export default Markdown
