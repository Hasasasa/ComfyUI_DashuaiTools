import { app } from "../../scripts/app.js";

// --- Dropdown UI Component ---
let activeDropdown = null;

class Dropdown {
    constructor(inputEl, options, onSelect, manualOffset, hostElement) {
        this.inputEl = inputEl;
        this.options = options;
        this.onSelect = onSelect;
        this.manualOffset = manualOffset;
        this.hostElement = hostElement;
        
        this.dropdown = document.createElement('ul');
        this.dropdown.className = 'dadashuai-dropdown';
        this.dropdown.setAttribute('role', 'listbox');
        
        // [Opt] 设置基础 Z-Index
        this.dropdown.style.zIndex = 10001;
        
        this.selectedIndex = -1;
        this.focusedDropdown = this.dropdown;
        this.openedAt = Date.now();

        this.buildDropdown();
        this.updatePosition();
        
        // Bind events
        this.handleKeyDown = this.onKeyDown.bind(this);
        this.handleWheel = this.onWheel.bind(this);
        this.handleClick = this.onClick.bind(this);
        
        document.addEventListener('keydown', this.handleKeyDown);
        this.dropdown.addEventListener('wheel', this.handleWheel);
        document.addEventListener('click', this.handleClick);
        
        activeDropdown = this;
    }

    buildDropdown() {
        this.buildNested(this.options, this.dropdown);
        this.hostElement.appendChild(this.dropdown);
    }

    buildNested(dict, parent, path = '') {
        let index = 0;
        // 计算当前菜单层级，用于关闭更深层的菜单
        const currentDepth = (path.split('/').filter(Boolean).length || 0) + 1;

        for (const [key, val] of Object.entries(dict)) {
            // [Fix] 这里的 idx 必须用 const 锁定当前循环的值
            const idx = index; 

            const extra = typeof val === 'string' ? val : '';
            let fullPath = path ? `${path}/${key}` : key;
            if (extra) fullPath += `###${extra}`;

            const li = document.createElement('li');
            
            if (val && typeof val === "object") {
                li.className = 'folder';
                li.textContent = key;
                
                const subUl = document.createElement('ul');
                subUl.className = 'dadashuai-nested-dropdown';
                subUl.dataset.depth = currentDepth + 1;
                
                // [Opt] 动态增加 Z-Index，确保子菜单永远覆盖父菜单
                subUl.style.zIndex = 10001 + currentDepth + 1;
                
                li.addEventListener('mouseenter', () => {
                    this.hideNested(subUl.dataset.depth, subUl);
                    this.positionNested(li, subUl);
                });
                li.addEventListener('mouseover', () => this.setFocus(idx, parent));
                
                // Prevent closing when clicking nested scrollbar
                subUl.addEventListener('mousedown', e => e.stopPropagation());
                subUl.addEventListener('click', e => e.stopPropagation());

                this.hostElement.appendChild(subUl);
                this.buildNested(val, subUl, fullPath);
            } else {
                li.className = 'item';
                li.setAttribute('role', 'option');
                li.textContent = key;
                
                // 鼠标滑过普通项时，关闭同级的子菜单
                li.addEventListener('mouseenter', () => {
                    this.hideNested(currentDepth + 1, null);
                });

                li.addEventListener('mouseover', () => this.setFocus(idx, parent));
                li.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    this.onSelect(key, fullPath);
                });
            }
            parent.appendChild(li);
            index++;
        }
    }

    updatePosition() {
        const rect = this.inputEl.getBoundingClientRect();
        const parse = (v, dim) => isNaN(v) && String(v).includes('%') ? dim * (parseInt(v)/100) : v;
        
        const offX = parse(this.manualOffset[0], rect.height);
        const offY = parse(this.manualOffset[1], rect.width);
        
        const winW = window.innerWidth;
        const winH = window.innerHeight;
        const margin = 10;

        // JS 中的 maxWidth 主要是为了初始计算，CSS 中的 max-width 才是强制生效的关键
        this.dropdown.style.maxWidth = `${Math.min(450, winW - 20)}px`;
        
        // Initial placement
        let top = rect.bottom - offX;
        let left = rect.right - offY;
        this.dropdown.style.top = `${top}px`;
        this.dropdown.style.left = `${left}px`;

        // Constraint to viewport
        const dropRect = this.dropdown.getBoundingClientRect();
        const newLeft = Math.max(margin, Math.min(dropRect.left, winW - dropRect.width - margin));
        const newTop = Math.max(margin, Math.min(dropRect.top, winH - dropRect.height - margin));
        
        this.dropdown.style.left = `${newLeft}px`;
        this.dropdown.style.top = `${newTop}px`;
    }

    positionNested(parentLi, subUl) {
        subUl.style.display = 'block';
        const pRect = parentLi.getBoundingClientRect();
        const sRect = subUl.getBoundingClientRect();
        const margin = 10;

        let left = pRect.right;
        if (left + sRect.width > window.innerWidth - margin) {
            left = pRect.left - sRect.width;
        }
        left = Math.max(margin, left);

        let top = pRect.top;
        if (top + sRect.height > window.innerHeight - margin) {
            top = Math.max(margin, window.innerHeight - sRect.height - margin);
        }

        subUl.style.top = `${top}px`;
        subUl.style.left = `${left}px`;
    }

    hideNested(depth, except) {
        document.querySelectorAll('.dadashuai-nested-dropdown').forEach(el => {
            if (Number(el.dataset.depth || 0) >= depth && el !== except) {
                el.style.display = 'none';
            }
        });
    }

    setFocus(index, parent) {
        if (parent) this.focusedDropdown = parent;
        this.selectedIndex = index;
        this.renderSelection();
    }

    renderSelection() {
        const items = Array.from(this.focusedDropdown.children);
        items.forEach((li, i) => li.classList.toggle('selected', i === this.selectedIndex));
    }

    destroy() {
        document.removeEventListener('keydown', this.handleKeyDown);
        this.dropdown.removeEventListener('wheel', this.handleWheel);
        document.removeEventListener('click', this.handleClick);
        document.querySelectorAll('.dadashuai-dropdown, .dadashuai-nested-dropdown').forEach(el => el.remove());
        activeDropdown = null;
    }

    onKeyDown(e) {
        if (!activeDropdown) return;
        const items = Array.from(this.focusedDropdown.children);
        const selected = items[this.selectedIndex];

        switch(e.key) {
            case 'ArrowUp':
                e.preventDefault();
                this.selectedIndex = Math.max(0, this.selectedIndex - 1);
                this.renderSelection();
                break;
            case 'ArrowDown':
                e.preventDefault();
                this.selectedIndex = Math.min(items.length - 1, this.selectedIndex + 1);
                this.renderSelection();
                break;
            case 'ArrowLeft':
                if (this.focusedDropdown !== this.dropdown) {
                    // Logic to go back up level could go here, simplified to just preventing default
                }
                break;
            case 'Enter':
            case 'Tab':
                if (this.selectedIndex >= 0 && selected?.classList.contains('item')) {
                    e.preventDefault();
                    // Note: Keyboard selection requires tracking fullPath which isn't stored in DOM in simplified version
                    // For now, click is primary. 
                    selected.dispatchEvent(new MouseEvent('mousedown'));
                }
                break;
            case 'Escape':
                this.destroy();
                break;
        }
    }

    onWheel(e) {
        const currentTop = parseInt(this.dropdown.style.top);
        const delta = e.deltaY < 0 ? 10 : -10;
        const invert = localStorage.getItem("Comfy.Settings.Comfy.InvertMenuScrolling");
        this.dropdown.style.top = `${currentTop + (invert ? -delta : delta)}px`;
    }

    onClick(e) {
        if (Date.now() - this.openedAt < 200) return;
        if (!e.target.closest('.dadashuai-dropdown, .dadashuai-nested-dropdown') && e.target !== this.inputEl) {
            this.destroy();
        }
    }
}

// --- Data & Logic Helpers ---

const IGNORE_WIDGETS = new Set(['control_after_generate', 'empty_latent_aspect', 'empty_latent_width', 'empty_latent_height', 'batch_size']);

function getNodeOptions(node) {
    if (!node.widgets) return null;
    const opts = {};
    
    for (const w of node.widgets) {
        if (!w.type || IGNORE_WIDGETS.has(w.name)) continue;
        
        const val = w.value;
        
        // Handle Seeds
        if (w.name === 'seed' || (w.name === 'value' && node.constructor.title?.toLowerCase() === 'seed')) {
            opts[w.name] = { [val ?? 'seed']: null };
            continue;
        }
        
        // Handle Toggles
        if (w.type === 'toggle') {
            opts[w.name] = { 'True': null, 'False': null };
            continue;
        }
        
        // Handle Text/Number inputs
        if (['customtext', 'text', 'string', 'number'].includes(w.type) || w.type.startsWith('dadashuai')) {
            const safeVal = (val !== undefined && val !== null && String(val).trim() !== '') ? val : 'value';
            opts[w.name] = { [safeVal]: null };
            continue;
        }
        
        // Handle Lists (Combo)
        if (w.options?.values) {
            let values = w.options.values;
            if (typeof values === 'function') values = values();
            const valDict = {};
            for (const v of values) valDict[v] = null;
            if (Object.keys(valDict).length > 0) opts[w.name] = valDict;
        }
    }
    return Object.keys(opts).length ? opts : null;
}

// [Fix] 增加参数 ignoreId 用于排除自身节点
function getGraphOptions(ignoreId) {
    const dict = {};
    for (const node of app.graph._nodes) {
        // [Fix] 排除当前节点自身
        if (node.id === ignoreId) continue;
        
        // [Fix] 排除没有输出端口的节点 (例如 Note 节点或纯输入节点)
        if (!node.outputs || node.outputs.length === 0) continue;

        const opts = getNodeOptions(node);
        if (opts) {
            const title = node.title || node.constructor?.title || node.type;
            dict[`[${node.id}] - ${title}`] = opts;
        }
    }
    return dict;
}

function insertValue(widget, fullpath, selectedVal) {
    // Parse path: "Node/Widget/Value"
    const [rawPath] = fullpath.split('###');
    const parts = rawPath.split('/');
    const nodeKey = parts[0] || '';
    const widgetName = parts[1] || '';
    let value = parts[2] ?? selectedVal;

    // Format value
    if (typeof value === 'boolean') value = value ? 'True' : 'False';
    const strVal = String(value);
    const finalVal = (strVal === 'True' || strVal === 'False' || (!isNaN(Number(strVal)) && strVal.trim() !== '')) 
        ? strVal 
        : `'${strVal.replace(/^['"]|['"]$/g, '')}'`; // Simple quote clean

    // Extract ID
    const nodeIdMatch = nodeKey.match(/\[(\d+)\]/);
    const nodeId = nodeIdMatch ? nodeIdMatch[1] : nodeKey;

    // Construct Line
    const line = `[${nodeId}:${widgetName}]=${finalVal}`;
    
    // Update Input
    const current = widget.inputEl.value || '';
    
    // Auto-increment index <N:name>
    const matches = current.match(/^<(\d+):/gm);
    let nextIdx = 1;
    if (matches) {
        const indices = matches.map(s => parseInt(s.substring(1)));
        nextIdx = Math.max(...indices) + 1;
    }

    // [Change] 自动生成标签: X1, X2... 或 Y1, Y2...
    const axisChar = widget.name === 'y_attr' ? 'Y' : 'X';
    const block = `<${nextIdx}:${axisChar}${nextIdx}>\n${line}`;
    
    widget.inputEl.value = current.trim() ? `${current.replace(/\s+$/,'')}\n${block}` : block;
    
    // Trigger update
    const ev = new Event('input', { bubbles: true });
    widget.inputEl.dispatchEvent(ev);
}

function injectStyles() {
    if (document.getElementById('dadashuai-style')) return;
    const style = document.createElement('style');
    style.id = 'dadashuai-style';
    style.textContent = `
        .dadashuai-dropdown, .dadashuai-nested-dropdown {
            position: fixed; background: #1c1c1c; color: #e8e8e8;
            border: 1px solid #363636; border-radius: 6px; padding: 0; /* [Fix] 移除容器内边距，消除顶部底部空隙 */
            list-style: none; max-height: 320px; min-width: 200px;
            max-width: 450px; /* [Fix] 限制最大宽度，强制长内容换行 */
            overflow-y: auto; z-index: 9999; font-size: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        /* [Fix] 将样式规则同时应用到 .dadashuai-nested-dropdown li */
        .dadashuai-dropdown li, .dadashuai-nested-dropdown li { 
            padding: 5px 12px; 
            cursor: pointer; 
            position: relative;
            /* [Fix] 强制换行策略：优先在单词间换行，必要时打断单词 */
            white-space: normal;
            word-wrap: break-word;
            overflow-wrap: break-word; 
            line-height: 1.4;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        /* [Fix] 移除最后一个子元素的底边框 */
        .dadashuai-dropdown li:last-child, .dadashuai-nested-dropdown li:last-child {
            border-bottom: none;
        }
        .dadashuai-dropdown li.selected, .dadashuai-nested-dropdown li.selected { background: #4a90e2; color: #fff; }
        .dadashuai-dropdown li.folder, .dadashuai-nested-dropdown li.folder { font-weight: 600; }
        .dadashuai-dropdown li.folder::after, .dadashuai-nested-dropdown li.folder::after { content: '›'; float: right; opacity: 0.6; }
        /* [Fix] 嵌套菜单必须是 fixed 定位以配合 getBoundingClientRect 计算 */
        .dadashuai-nested-dropdown { display: none; position: fixed; }
    `;
    document.head.appendChild(style);
}

function attachDropdown(node) {
    if (!node.widgets) return;
    injectStyles();
    
    const targets = node.widgets.filter(w => ['x_attr', 'y_attr'].includes(w.name));
    
    for (const w of targets) {
        const handler = (e) => {
            // Only show if widget has input element and data exists
            if (!w.inputEl) return;
            
            // [Fix] 传递当前节点的 ID，以便在获取选项时排除自身
            const data = getGraphOptions(node.id);
            if (!data || Object.keys(data).length === 0) return;

            if (activeDropdown) activeDropdown.destroy();
            
            // Create new dropdown
            new Dropdown(w.inputEl, data, (val, path) => insertValue(w, path, val), [10, '100%'], document.body);
        };

        // Retry attachment if DOM not ready
        const tryAttach = () => {
            if (!w.inputEl) return false;
            ['click', 'focus'].forEach(evt => w.inputEl.addEventListener(evt, handler));
            return true;
        };

        if (!tryAttach()) {
            const timer = setInterval(() => {
                if (tryAttach()) clearInterval(timer);
            }, 100);
        }
    }
}

app.registerExtension({
    name: "comfy.xyd.dropdown",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "XY_Image" && nodeData.widgets) {
            nodeData.widgets.forEach(w => {
                if (['x_attr', 'y_attr'].includes(w.name)) {
                    w.type = 'customtext';
                    w.options = { ...w.options, multiline: true };
                }
            });
        }
    },
    nodeCreated(node) {
        if (node.constructor.type === "XY_Image" || node.title === "XY_Image") {
            attachDropdown(node);
        }
    }
});