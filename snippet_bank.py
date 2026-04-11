"""
Snippet Bank — Procedural Episode Generator for CodeReviewEnv

Contains 35+ clean code snippets across Python, JavaScript, and Go,
plus 5 bug injectors that mutate clean code into buggy code.

Every reset() draws a fresh snippet, applies 1-3 random injectors,
and stores the gold answer in State. No static dataset — infinite
unique episodes from a finite snippet bank.

Injectors:
  1. off_by_one       — flips < to <=, range(n) to range(n-1), etc.
  2. null_deref       — removes a None/null/nil guard
  3. wrong_operator   — swaps + for -, * for /, and↔or
  4. unused_var       — inserts a dead variable shadowing a live one
  5. logic_inversion  — flips True/False, ==/!=, and/or
"""

import ast
import random
import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class Snippet:
    """A clean, correct code snippet that can have bugs injected."""
    name: str
    language: str       # python | javascript | go
    difficulty: str     # easy | medium | hard
    code: str
    description: str


@dataclass
class BugRecord:
    """Record of an injected bug — stored in State as gold answer."""
    description: str
    lines: List[int]
    fix: str
    bug_type: str


# ─── Snippet Bank ────────────────────────────────────────────────────────────

SNIPPET_BANK: List[Snippet] = [
    # ── Python — Easy ────────────────────────────────────────────────
    Snippet(
        name="binary_search",
        language="python",
        difficulty="easy",
        code="""\
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""",
        description="Binary search in a sorted array",
    ),
    Snippet(
        name="fibonacci",
        language="python",
        difficulty="easy",
        code="""\
def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
""",
        description="Compute nth Fibonacci number iteratively",
    ),
    Snippet(
        name="max_subarray",
        language="python",
        difficulty="easy",
        code="""\
def max_subarray(nums):
    if not nums:
        return 0
    current_sum = max_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum
""",
        description="Kadane's algorithm for maximum subarray sum",
    ),
    Snippet(
        name="is_palindrome",
        language="python",
        difficulty="easy",
        code="""\
def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    left, right = 0, len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    return True
""",
        description="Check if string is a palindrome",
    ),
    Snippet(
        name="reverse_linked_list",
        language="python",
        difficulty="easy",
        code="""\
def reverse_linked_list(head):
    prev = None
    current = head
    while current is not None:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
""",
        description="Reverse a singly linked list in place",
    ),
    # ── Python — Medium ──────────────────────────────────────────────
    Snippet(
        name="merge_sort",
        language="python",
        difficulty="medium",
        code="""\
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""",
        description="Merge sort with stable ordering",
    ),
    Snippet(
        name="lru_cache",
        language="python",
        difficulty="medium",
        code="""\
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key):
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
""",
        description="LRU cache with get/put operations",
    ),
    Snippet(
        name="flatten_dict",
        language="python",
        difficulty="medium",
        code="""\
def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict) and v:
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items
""",
        description="Flatten a nested dictionary with dot-separated keys",
    ),
    Snippet(
        name="validate_email",
        language="python",
        difficulty="medium",
        code="""\
def validate_email(email):
    if not email or not isinstance(email, str):
        return False
    parts = email.split('@')
    if len(parts) != 2:
        return False
    local, domain = parts
    if not local or not domain:
        return False
    if '.' not in domain:
        return False
    if domain.startswith('.') or domain.endswith('.'):
        return False
    if '..' in domain:
        return False
    return True
""",
        description="Basic email validation without regex",
    ),
    Snippet(
        name="dijkstra",
        language="python",
        difficulty="medium",
        code="""\
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    while pq:
        dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return distances
""",
        description="Dijkstra's shortest path algorithm",
    ),
    # ── Python — Hard ────────────────────────────────────────────────
    Snippet(
        name="rate_limiter",
        language="python",
        difficulty="hard",
        code="""\
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_id):
        now = time.time()
        cutoff = now - self.window
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > cutoff
        ]
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True

    def remaining(self, client_id):
        now = time.time()
        cutoff = now - self.window
        active = [t for t in self.requests[client_id] if t > cutoff]
        return max(0, self.max_requests - len(active))
""",
        description="Token bucket rate limiter with sliding window",
    ),
    Snippet(
        name="trie",
        language="python",
        difficulty="hard",
        code="""\
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
""",
        description="Trie (prefix tree) with insert, search, starts_with",
    ),
    Snippet(
        name="json_parser",
        language="python",
        difficulty="hard",
        code="""\
def parse_json_value(s, pos):
    if pos >= len(s):
        raise ValueError("Unexpected end of input")
    ch = s[pos]
    if ch == '"':
        return parse_string(s, pos)
    if ch == '{':
        return parse_object(s, pos)
    if ch == '[':
        return parse_array(s, pos)
    if ch in '-0123456789':
        return parse_number(s, pos)
    if s[pos:pos+4] == 'true':
        return True, pos + 4
    if s[pos:pos+5] == 'false':
        return False, pos + 5
    if s[pos:pos+4] == 'null':
        return None, pos + 4
    raise ValueError(f"Unexpected character at {pos}: {ch}")

def parse_string(s, pos):
    assert s[pos] == '"'
    pos += 1
    result = []
    while pos < len(s) and s[pos] != '"':
        if s[pos] == '\\\\':
            pos += 1
            result.append(s[pos])
        else:
            result.append(s[pos])
        pos += 1
    return ''.join(result), pos + 1

def parse_number(s, pos):
    start = pos
    if s[pos] == '-':
        pos += 1
    while pos < len(s) and s[pos].isdigit():
        pos += 1
    if pos < len(s) and s[pos] == '.':
        pos += 1
        while pos < len(s) and s[pos].isdigit():
            pos += 1
    return float(s[start:pos]), pos

def parse_array(s, pos):
    assert s[pos] == '['
    pos += 1
    result = []
    pos = skip_whitespace(s, pos)
    if pos < len(s) and s[pos] == ']':
        return result, pos + 1
    while True:
        pos = skip_whitespace(s, pos)
        value, pos = parse_json_value(s, pos)
        result.append(value)
        pos = skip_whitespace(s, pos)
        if pos < len(s) and s[pos] == ',':
            pos += 1
        else:
            break
    assert s[pos] == ']'
    return result, pos + 1

def parse_object(s, pos):
    assert s[pos] == '{'
    pos += 1
    result = {}
    pos = skip_whitespace(s, pos)
    if pos < len(s) and s[pos] == '}':
        return result, pos + 1
    while True:
        pos = skip_whitespace(s, pos)
        key, pos = parse_string(s, pos)
        pos = skip_whitespace(s, pos)
        assert s[pos] == ':'
        pos += 1
        pos = skip_whitespace(s, pos)
        value, pos = parse_json_value(s, pos)
        result[key] = value
        pos = skip_whitespace(s, pos)
        if pos < len(s) and s[pos] == ',':
            pos += 1
        else:
            break
    assert s[pos] == '}'
    return result, pos + 1

def skip_whitespace(s, pos):
    while pos < len(s) and s[pos] in ' \\t\\n\\r':
        pos += 1
    return pos
""",
        description="Recursive descent JSON parser",
    ),
    Snippet(
        name="thread_pool",
        language="python",
        difficulty="hard",
        code="""\
import threading
from collections import deque

class ThreadPool:
    def __init__(self, num_workers):
        self.tasks = deque()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.shutdown_flag = False
        self.workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def submit(self, func, *args):
        with self.condition:
            if self.shutdown_flag:
                raise RuntimeError("Pool is shut down")
            self.tasks.append((func, args))
            self.condition.notify()

    def _worker(self):
        while True:
            with self.condition:
                while not self.tasks and not self.shutdown_flag:
                    self.condition.wait()
                if self.shutdown_flag and not self.tasks:
                    return
                func, args = self.tasks.popleft()
            func(*args)

    def shutdown(self):
        with self.condition:
            self.shutdown_flag = True
            self.condition.notify_all()
        for w in self.workers:
            w.join()
""",
        description="Simple thread pool with task queue and graceful shutdown",
    ),
    Snippet(
        name="auth_handler",
        language="python",
        difficulty="hard",
        code="""\
import hashlib
import hmac
import time

TOKENS = {}

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()

def create_token(user_id, secret, expires_in=3600):
    payload = f"{user_id}:{int(time.time()) + expires_in}"
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    token = f"{payload}:{signature}"
    TOKENS[token] = user_id
    return token

def validate_token(token, secret):
    if not token or ':' not in token:
        return None
    parts = token.rsplit(':', 1)
    if len(parts) != 2:
        return None
    payload, signature = parts
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    user_id, expires = payload.split(':', 1)
    if int(expires) < int(time.time()):
        return None
    return user_id
""",
        description="Token-based authentication with HMAC signing",
    ),

    # ── JavaScript — Easy ────────────────────────────────────────────
    Snippet(
        name="array_flatten",
        language="javascript",
        difficulty="easy",
        code="""\
function flatten(arr) {
    const result = [];
    for (let i = 0; i < arr.length; i++) {
        if (Array.isArray(arr[i])) {
            const nested = flatten(arr[i]);
            for (let j = 0; j < nested.length; j++) {
                result.push(nested[j]);
            }
        } else {
            result.push(arr[i]);
        }
    }
    return result;
}
""",
        description="Recursively flatten a nested array",
    ),
    Snippet(
        name="debounce",
        language="javascript",
        difficulty="easy",
        code="""\
function debounce(func, delay) {
    let timer = null;
    return function(...args) {
        if (timer !== null) {
            clearTimeout(timer);
        }
        timer = setTimeout(() => {
            func.apply(this, args);
            timer = null;
        }, delay);
    };
}
""",
        description="Debounce function for rate-limiting calls",
    ),
    Snippet(
        name="deep_clone",
        language="javascript",
        difficulty="easy",
        code="""\
function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    if (Array.isArray(obj)) {
        return obj.map(item => deepClone(item));
    }
    const clone = {};
    for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
            clone[key] = deepClone(obj[key]);
        }
    }
    return clone;
}
""",
        description="Deep clone an object without JSON.parse",
    ),
    Snippet(
        name="event_emitter",
        language="javascript",
        difficulty="easy",
        code="""\
class EventEmitter {
    constructor() {
        this.listeners = {};
    }
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }
    off(event, callback) {
        if (!this.listeners[event]) return;
        this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }
    emit(event, ...args) {
        if (!this.listeners[event]) return;
        for (const cb of this.listeners[event]) {
            cb(...args);
        }
    }
}
""",
        description="Simple event emitter with on/off/emit",
    ),

    # ── JavaScript — Medium ──────────────────────────────────────────
    Snippet(
        name="promise_all",
        language="javascript",
        difficulty="medium",
        code="""\
function promiseAll(promises) {
    return new Promise((resolve, reject) => {
        if (promises.length === 0) {
            resolve([]);
            return;
        }
        const results = new Array(promises.length);
        let completed = 0;
        for (let i = 0; i < promises.length; i++) {
            Promise.resolve(promises[i]).then(value => {
                results[i] = value;
                completed += 1;
                if (completed === promises.length) {
                    resolve(results);
                }
            }).catch(reject);
        }
    });
}
""",
        description="Implement Promise.all from scratch",
    ),
    Snippet(
        name="throttle",
        language="javascript",
        difficulty="medium",
        code="""\
function throttle(func, limit) {
    let lastRun = 0;
    let timer = null;
    return function(...args) {
        const now = Date.now();
        const remaining = limit - (now - lastRun);
        if (remaining <= 0) {
            if (timer !== null) {
                clearTimeout(timer);
                timer = null;
            }
            lastRun = now;
            func.apply(this, args);
        } else if (timer === null) {
            timer = setTimeout(() => {
                lastRun = Date.now();
                timer = null;
                func.apply(this, args);
            }, remaining);
        }
    };
}
""",
        description="Throttle function with trailing call",
    ),
    Snippet(
        name="virtual_dom_diff",
        language="javascript",
        difficulty="medium",
        code="""\
function diff(oldNode, newNode) {
    if (oldNode === null) {
        return { type: 'CREATE', node: newNode };
    }
    if (newNode === null) {
        return { type: 'REMOVE' };
    }
    if (typeof oldNode !== typeof newNode) {
        return { type: 'REPLACE', node: newNode };
    }
    if (typeof oldNode === 'string') {
        if (oldNode !== newNode) {
            return { type: 'REPLACE', node: newNode };
        }
        return null;
    }
    if (oldNode.tag !== newNode.tag) {
        return { type: 'REPLACE', node: newNode };
    }
    const childPatches = [];
    const maxLen = Math.max(
        oldNode.children.length,
        newNode.children.length
    );
    for (let i = 0; i < maxLen; i++) {
        const patch = diff(
            oldNode.children[i] || null,
            newNode.children[i] || null
        );
        childPatches.push(patch);
    }
    return { type: 'UPDATE', children: childPatches };
}
""",
        description="Virtual DOM diff algorithm",
    ),
    Snippet(
        name="middleware_chain",
        language="javascript",
        difficulty="medium",
        code="""\
function createMiddlewareChain(middlewares) {
    return function(req, res) {
        let index = 0;
        function next(err) {
            if (err) {
                res.status(500).send(err.message);
                return;
            }
            if (index >= middlewares.length) {
                return;
            }
            const middleware = middlewares[index];
            index += 1;
            try {
                middleware(req, res, next);
            } catch (e) {
                next(e);
            }
        }
        next();
    };
}
""",
        description="Express-style middleware chain executor",
    ),

    # ── JavaScript — Hard ────────────────────────────────────────────
    Snippet(
        name="reactive_state",
        language="javascript",
        difficulty="hard",
        code="""\
function createReactiveState(initialState) {
    const subscribers = new Map();
    let state = { ...initialState };

    function subscribe(key, callback) {
        if (!subscribers.has(key)) {
            subscribers.set(key, new Set());
        }
        subscribers.get(key).add(callback);
        return () => subscribers.get(key).delete(callback);
    }

    function setState(updates) {
        const changed = [];
        for (const [key, value] of Object.entries(updates)) {
            if (state[key] !== value) {
                state[key] = value;
                changed.push(key);
            }
        }
        for (const key of changed) {
            if (subscribers.has(key)) {
                for (const cb of subscribers.get(key)) {
                    cb(state[key], key);
                }
            }
        }
    }

    function getState(key) {
        if (key !== undefined) {
            return state[key];
        }
        return { ...state };
    }

    return { subscribe, setState, getState };
}
""",
        description="Reactive state management with subscriptions",
    ),

    # ── Go — Easy ────────────────────────────────────────────────────
    Snippet(
        name="stack",
        language="go",
        difficulty="easy",
        code="""\
package stack

type Stack struct {
    items []int
}

func (s *Stack) Push(item int) {
    s.items = append(s.items, item)
}

func (s *Stack) Pop() (int, bool) {
    if len(s.items) == 0 {
        return 0, false
    }
    last := len(s.items) - 1
    item := s.items[last]
    s.items = s.items[:last]
    return item, true
}

func (s *Stack) Peek() (int, bool) {
    if len(s.items) == 0 {
        return 0, false
    }
    return s.items[len(s.items)-1], true
}

func (s *Stack) Size() int {
    return len(s.items)
}
""",
        description="Generic integer stack with push/pop/peek",
    ),
    Snippet(
        name="string_reverse",
        language="go",
        difficulty="easy",
        code="""\
package stringutil

func Reverse(s string) string {
    runes := []rune(s)
    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
        runes[i], runes[j] = runes[j], runes[i]
    }
    return string(runes)
}

func IsPalindrome(s string) bool {
    runes := []rune(s)
    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
        if runes[i] != runes[j] {
            return false
        }
    }
    return true
}
""",
        description="String reversal and palindrome check in Go",
    ),
    Snippet(
        name="go_binary_search",
        language="go",
        difficulty="easy",
        code="""\
package search

func BinarySearch(arr []int, target int) int {
    low, high := 0, len(arr)-1
    for low <= high {
        mid := low + (high-low)/2
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            low = mid + 1
        } else {
            high = mid - 1
        }
    }
    return -1
}
""",
        description="Binary search returning index or -1",
    ),

    # ── Go — Medium ──────────────────────────────────────────────────
    Snippet(
        name="concurrent_map",
        language="go",
        difficulty="medium",
        code="""\
package safemap

import "sync"

type SafeMap struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func NewSafeMap() *SafeMap {
    return &SafeMap{data: make(map[string]interface{})}
}

func (m *SafeMap) Get(key string) (interface{}, bool) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    val, ok := m.data[key]
    return val, ok
}

func (m *SafeMap) Set(key string, value interface{}) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.data[key] = value
}

func (m *SafeMap) Delete(key string) {
    m.mu.Lock()
    defer m.mu.Unlock()
    delete(m.data, key)
}

func (m *SafeMap) Len() int {
    m.mu.RLock()
    defer m.mu.RUnlock()
    return len(m.data)
}
""",
        description="Thread-safe map with RWMutex",
    ),
    Snippet(
        name="go_linked_list",
        language="go",
        difficulty="medium",
        code="""\
package linkedlist

type Node struct {
    Value int
    Next  *Node
}

type LinkedList struct {
    Head *Node
    Size int
}

func (ll *LinkedList) Append(val int) {
    newNode := &Node{Value: val}
    if ll.Head == nil {
        ll.Head = newNode
        ll.Size++
        return
    }
    current := ll.Head
    for current.Next != nil {
        current = current.Next
    }
    current.Next = newNode
    ll.Size++
}

func (ll *LinkedList) Remove(val int) bool {
    if ll.Head == nil {
        return false
    }
    if ll.Head.Value == val {
        ll.Head = ll.Head.Next
        ll.Size--
        return true
    }
    current := ll.Head
    for current.Next != nil {
        if current.Next.Value == val {
            current.Next = current.Next.Next
            ll.Size--
            return true
        }
        current = current.Next
    }
    return false
}
""",
        description="Singly linked list with append and remove",
    ),
    Snippet(
        name="go_worker_pool",
        language="go",
        difficulty="medium",
        code="""\
package workerpool

import "sync"

type Task func()

type Pool struct {
    tasks   chan Task
    wg      sync.WaitGroup
    workers int
}

func NewPool(workers, queueSize int) *Pool {
    p := &Pool{
        tasks:   make(chan Task, queueSize),
        workers: workers,
    }
    for i := 0; i < workers; i++ {
        go p.worker()
    }
    return p
}

func (p *Pool) worker() {
    for task := range p.tasks {
        task()
        p.wg.Done()
    }
}

func (p *Pool) Submit(task Task) {
    p.wg.Add(1)
    p.tasks <- task
}

func (p *Pool) Wait() {
    p.wg.Wait()
}

func (p *Pool) Close() {
    close(p.tasks)
}
""",
        description="Worker pool with goroutines and WaitGroup",
    ),

    # ── Go — Hard ────────────────────────────────────────────────────
    Snippet(
        name="go_channel_pipeline",
        language="go",
        difficulty="hard",
        code="""\
package pipeline

func Generator(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

func Filter(in <-chan int, predicate func(int) bool) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            if predicate(n) {
                out <- n
            }
        }
        close(out)
    }()
    return out
}

func Map(in <-chan int, transform func(int) int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- transform(n)
        }
        close(out)
    }()
    return out
}

func Reduce(in <-chan int, initial int, combine func(int, int) int) int {
    result := initial
    for n := range in {
        result = combine(result, n)
    }
    return result
}
""",
        description="Channel-based pipeline with generator/filter/map/reduce",
    ),

    # ── More Python snippets for variety ─────────────────────────────
    Snippet(
        name="csv_parser",
        language="python",
        difficulty="medium",
        code="""\
def parse_csv(text, delimiter=','):
    rows = []
    for line in text.strip().split('\\n'):
        fields = []
        current = ''
        in_quotes = False
        for ch in line:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == delimiter and not in_quotes:
                fields.append(current.strip())
                current = ''
            else:
                current += ch
        fields.append(current.strip())
        rows.append(fields)
    return rows
""",
        description="CSV parser handling quoted fields",
    ),
    Snippet(
        name="topological_sort",
        language="python",
        difficulty="hard",
        code="""\
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    queue = [node for node in in_degree if in_degree[node] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(result) != len(in_degree):
        raise ValueError("Graph contains a cycle")
    return result
""",
        description="Kahn's algorithm for topological sorting",
    ),
    Snippet(
        name="connection_pool",
        language="python",
        difficulty="hard",
        code="""\
import threading
import time

class ConnectionPool:
    def __init__(self, max_size, factory):
        self.max_size = max_size
        self.factory = factory
        self.pool = []
        self.in_use = set()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def acquire(self, timeout=None):
        deadline = time.time() + timeout if timeout else None
        with self.condition:
            while True:
                if self.pool:
                    conn = self.pool.pop()
                    self.in_use.add(id(conn))
                    return conn
                if len(self.in_use) < self.max_size:
                    conn = self.factory()
                    self.in_use.add(id(conn))
                    return conn
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError("Connection pool exhausted")
                    self.condition.wait(remaining)
                else:
                    self.condition.wait()

    def release(self, conn):
        with self.condition:
            self.in_use.discard(id(conn))
            self.pool.append(conn)
            self.condition.notify()
""",
        description="Thread-safe connection pool with timeout",
    ),
    Snippet(
        name="group_by",
        language="python",
        difficulty="easy",
        code="""\
def group_by(items, key_func):
    groups = {}
    for item in items:
        key = key_func(item)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups

def count_by(items, key_func):
    counts = {}
    for item in items:
        key = key_func(item)
        counts[key] = counts.get(key, 0) + 1
    return counts
""",
        description="Group and count items by a key function",
    ),
    Snippet(
        name="matrix_multiply",
        language="python",
        difficulty="easy",
        code="""\
def matrix_multiply(a, b):
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("Incompatible matrix dimensions")
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result
""",
        description="Matrix multiplication with dimension validation",
    ),
]


# ─── Bug Injectors ──────────────────────────────────────────────────────────
# Each injector: (code, rng) -> (buggy_code, BugRecord)
# Returns None if no applicable injection site found.


def off_by_one_injector(code: str, rng: random.Random) -> Optional[Tuple[str, BugRecord]]:
    """Introduce an off-by-one error: flip < to <=, > to >=, or adjust range bounds."""
    lines = code.split('\n')
    candidates = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('//'):
            continue
        # Pattern: while/for/if with < that could become <=
        if re.search(r'[^<>=!]\s*<\s*[^<=]', line) and ('while' in line or 'for' in line or 'if' in line):
            candidates.append((i, 'lt_to_lte', line))
        # Pattern: <= that could become <
        if re.search(r'<=', line) and ('while' in line or 'for' in line or 'if' in line):
            candidates.append((i, 'lte_to_lt', line))
        # Pattern: range(n) -> range(n-1) or range(n+1)
        if 'range(' in line:
            candidates.append((i, 'range_off', line))
        # Pattern: len(x) - 1 -> len(x) or len(x) - 2
        if re.search(r'len\([^)]+\)\s*-\s*1', line):
            candidates.append((i, 'len_off', line))

    if not candidates:
        return None

    idx, pattern, original_line = rng.choice(candidates)

    if pattern == 'lt_to_lte':
        new_line = re.sub(r'([^<>=!])\s*<\s*([^<=])', r'\1 <= \2', original_line, count=1)
        fix = "Change <= back to < to fix the off-by-one boundary"
    elif pattern == 'lte_to_lt':
        new_line = original_line.replace('<=', '<', 1)
        fix = "Change < back to <= to include the boundary value"
    elif pattern == 'range_off':
        m = re.search(r'range\(([^,)]+)\)', original_line)
        if m:
            arg = m.group(1).strip()
            new_line = original_line.replace(f'range({arg})', f'range({arg} - 1)', 1)
            fix = f"Change range({arg} - 1) back to range({arg}) to include last element"
        else:
            return None
    elif pattern == 'len_off':
        new_line = re.sub(r'len\(([^)]+)\)\s*-\s*1', r'len(\1)', original_line, count=1)
        fix = "Restore `- 1` after len() to avoid index out of bounds"
    else:
        return None

    if new_line == original_line:
        return None

    lines[idx] = new_line
    return '\n'.join(lines), BugRecord(
        description=f"Off-by-one error on line {idx + 1}: boundary condition is wrong",
        lines=[idx + 1],
        fix=fix,
        bug_type="off_by_one",
    )


def null_deref_injector(code: str, rng: random.Random) -> Optional[Tuple[str, BugRecord]]:
    """Remove a null/None/nil guard, causing a potential null dereference."""
    lines = code.split('\n')
    candidates = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Python: if x is not None, if x is None, if not x, if x
        if re.search(r'if\s+\w+\s+is\s+not\s+None', stripped):
            candidates.append((i, 'python_guard'))
        elif re.search(r'if\s+not\s+\w+', stripped) and 'return' not in stripped:
            candidates.append((i, 'python_not_guard'))
        # JS: if (x !== null), if (x != null), if (x)
        if re.search(r'if\s*\(\s*\w+\s*!==?\s*null\s*\)', stripped):
            candidates.append((i, 'js_guard'))
        # Go: if x != nil
        if re.search(r'if\s+\w+\s*!=\s*nil', stripped):
            candidates.append((i, 'go_guard'))
        # Python: if len(x) == 0 / if not x
        if re.search(r'if\s+(not\s+\w+|len\(\w+\)\s*==\s*0)', stripped):
            candidates.append((i, 'empty_guard'))

    if not candidates:
        return None

    idx, pattern = rng.choice(candidates)
    original_line = lines[idx]
    indent = len(original_line) - len(original_line.lstrip())

    # Find the body of the guard (next indented lines)
    body_lines = []
    for j in range(idx + 1, min(idx + 5, len(lines))):
        if lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) > indent:
            body_lines.append(j)
        else:
            break

    if body_lines:
        # Remove the guard but keep the body (de-indent by one level)
        body_indent = len(lines[body_lines[0]]) - len(lines[body_lines[0]].lstrip())
        dedent = body_indent - indent

        new_lines = list(lines)
        # Delete the guard line entirely (shift body up)
        new_lines.pop(idx)
        # Adjust body indices after removal
        for j_offset, j in enumerate(body_lines):
            adjusted_j = j - 1  # shifted up by one
            if adjusted_j < len(new_lines) and dedent > 0:
                new_lines[adjusted_j] = lines[j][dedent:]
    else:
        # Delete the guard line entirely
        new_lines = list(lines)
        new_lines.pop(idx)

    return '\n'.join(new_lines), BugRecord(
        description=f"Null/None guard removed on line {idx + 1}: missing null check before dereference",
        lines=[idx + 1],
        fix=f"Restore the null guard: {original_line.strip()}",
        bug_type="null_deref",
    )


def wrong_operator_injector(code: str, rng: random.Random) -> Optional[Tuple[str, BugRecord]]:
    """Swap an arithmetic or comparison operator: + ↔ -, * ↔ /, == ↔ !=."""
    lines = code.split('\n')
    candidates = []

    swaps = [
        (r'(\w)\s*\+\s*(\w)', r'\1 - \2', '+', '-'),
        (r'(\w)\s*-\s*(\w)', r'\1 + \2', '-', '+'),
        (r'(\w)\s*\*\s*(\w)', r'\1 / \2', '*', '/'),
    ]

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('//'):
            continue
        # Don't mess with string concatenation or imports
        if '"' in line or "'" in line or 'import' in line:
            continue
        for pattern, replacement, old_op, new_op in swaps:
            if re.search(pattern, line):
                candidates.append((i, pattern, replacement, old_op, new_op))

    if not candidates:
        return None

    idx, pattern, replacement, old_op, new_op = rng.choice(candidates)
    new_line = re.sub(pattern, replacement, lines[idx], count=1)

    if new_line == lines[idx]:
        return None

    new_lines = list(lines)
    new_lines[idx] = new_line

    return '\n'.join(new_lines), BugRecord(
        description=f"Wrong operator on line {idx + 1}: '{old_op}' should be '{new_op}'",
        lines=[idx + 1],
        fix=f"Change '{new_op}' back to '{old_op}' on line {idx + 1}",
        bug_type="wrong_operator",
    )


def unused_var_injector(code: str, rng: random.Random) -> Optional[Tuple[str, BugRecord]]:
    """Insert a dead variable that shadows a live one, causing subtle bugs."""
    lines = code.split('\n')

    # Find variable assignments to shadow
    assignments = []
    for i, line in enumerate(lines):
        # Python: x = ...
        m = re.match(r'^(\s+)(\w+)\s*=\s*', line)
        if m and not line.strip().startswith('#') and not line.strip().startswith('def '):
            indent = m.group(1)
            var_name = m.group(2)
            if var_name not in ('self', 'cls', 'result', 'return') and len(var_name) > 1:
                assignments.append((i, indent, var_name))
        # JS: let/const/var x = ...
        m = re.match(r'^(\s+)(?:let|const|var)\s+(\w+)\s*=', line)
        if m:
            indent = m.group(1)
            var_name = m.group(2)
            assignments.append((i, indent, var_name))

    if not assignments:
        return None

    idx, indent, var_name = rng.choice(assignments)

    # Insert a shadowing assignment before the real one
    shadow_values = ['0', 'None', '""', '[]', 'False', '{}']
    shadow_val = rng.choice(shadow_values)
    shadow_line = f"{indent}{var_name} = {shadow_val}"

    new_lines = list(lines)
    new_lines.insert(idx, shadow_line)

    return '\n'.join(new_lines), BugRecord(
        description=f"Dead variable on line {idx + 1}: '{var_name}' is assigned {shadow_val} but immediately overwritten — may mask intent or cause bugs if reordered",
        lines=[idx + 1],
        fix=f"Remove the dead assignment `{var_name} = {shadow_val}` on line {idx + 1}",
        bug_type="unused_var",
    )


def logic_inversion_injector(code: str, rng: random.Random) -> Optional[Tuple[str, BugRecord]]:
    """Flip a boolean: True↔False, and↔or, ==↔!=."""
    lines = code.split('\n')
    candidates = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('//'):
            continue
        if ' and ' in line:
            candidates.append((i, 'and_to_or'))
        if ' or ' in line and 'import' not in line:
            candidates.append((i, 'or_to_and'))
        if ' == ' in line and '==' not in stripped[:3]:
            candidates.append((i, 'eq_to_neq'))
        if ' != ' in line:
            candidates.append((i, 'neq_to_eq'))
        if 'True' in line and 'return True' in stripped:
            candidates.append((i, 'true_to_false'))
        if 'False' in line and 'return False' in stripped:
            candidates.append((i, 'false_to_true'))
        # JS/Go: true/false
        if 'true' in line and 'return true' in stripped:
            candidates.append((i, 'true_to_false'))
        if 'false' in line and 'return false' in stripped:
            candidates.append((i, 'false_to_true'))

    if not candidates:
        return None

    idx, pattern = rng.choice(candidates)
    original_line = lines[idx]

    if pattern == 'and_to_or':
        new_line = original_line.replace(' and ', ' or ', 1)
        fix = "Change 'or' back to 'and'"
    elif pattern == 'or_to_and':
        new_line = original_line.replace(' or ', ' and ', 1)
        fix = "Change 'and' back to 'or'"
    elif pattern == 'eq_to_neq':
        new_line = original_line.replace(' == ', ' != ', 1)
        fix = "Change '!=' back to '=='"
    elif pattern == 'neq_to_eq':
        new_line = original_line.replace(' != ', ' == ', 1)
        fix = "Change '==' back to '!='"
    elif pattern == 'true_to_false':
        new_line = original_line.replace('True', 'False', 1).replace('true', 'false', 1)
        fix = "Change 'False' back to 'True'"
    elif pattern == 'false_to_true':
        new_line = original_line.replace('False', 'True', 1).replace('false', 'true', 1)
        fix = "Change 'True' back to 'False'"
    else:
        return None

    if new_line == original_line:
        return None

    new_lines = list(lines)
    new_lines[idx] = new_line

    return '\n'.join(new_lines), BugRecord(
        description=f"Logic inversion on line {idx + 1}: boolean condition is flipped",
        lines=[idx + 1],
        fix=fix,
        bug_type="logic_inversion",
    )


# ─── AST-Based Injectors (Python only) ──────────────────────────────────────
# These use the ast module to find injection sites with structural accuracy,
# then apply the mutation via string replacement to preserve formatting.


def _is_python(code: str) -> bool:
    """Check if code parses as valid Python."""
    try:
        ast.parse(textwrap.dedent(code))
        return True
    except SyntaxError:
        return False


def ast_comparison_flip_injector(code: str, rng: random.Random) -> Optional[Tuple[str, BugRecord]]:
    """AST-based: find comparison operators and flip them (< ↔ <=, == ↔ !=, > ↔ >=)."""
    if not _is_python(code):
        return None

    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return None

    # Collect all Compare nodes with their line numbers
    flips = {
        ast.Lt: (ast.LtE, '<', '<=', "Change '<=' back to '<'"),
        ast.LtE: (ast.Lt, '<=', '<', "Change '<' back to '<='"),
        ast.Gt: (ast.GtE, '>', '>=', "Change '>=' back to '>'"),
        ast.GtE: (ast.Gt, '>=', '>', "Change '>' back to '>='"),
        ast.Eq: (ast.NotEq, '==', '!=', "Change '!=' back to '=='"),
        ast.NotEq: (ast.Eq, '!=', '==', "Change '==' back to '!='"),
    }

    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in flips:
                    candidates.append((node.lineno, type(op)))

    if not candidates:
        return None

    lineno, op_type = rng.choice(candidates)
    new_op_cls, old_str, new_str, fix = flips[op_type]

    lines = code.split('\n')
    if lineno - 1 >= len(lines):
        return None

    original_line = lines[lineno - 1]
    # Apply the string-level replacement on the target line
    new_line = original_line.replace(f' {old_str} ', f' {new_str} ', 1)
    if new_line == original_line:
        return None

    lines[lineno - 1] = new_line
    return '\n'.join(lines), BugRecord(
        description=f"Comparison operator flipped on line {lineno}: '{old_str}' changed to '{new_str}'",
        lines=[lineno],
        fix=fix,
        bug_type="ast_comparison_flip",
    )


def ast_binop_swap_injector(code: str, rng: random.Random) -> Optional[Tuple[str, BugRecord]]:
    """AST-based: find binary operations and swap operators (+ ↔ -, * ↔ //)."""
    if not _is_python(code):
        return None

    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return None

    swaps = {
        ast.Add: (ast.Sub, '+', '-', "Change '-' back to '+'"),
        ast.Sub: (ast.Add, '-', '+', "Change '+' back to '-'"),
        ast.Mult: (ast.FloorDiv, '*', '//', "Change '//' back to '*'"),
        ast.FloorDiv: (ast.Mult, '//', '*', "Change '*' back to '//'"),
    }

    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in swaps:
            candidates.append((node.lineno, type(node.op)))

    if not candidates:
        return None

    lineno, op_type = rng.choice(candidates)
    _, old_str, new_str, fix = swaps[op_type]

    lines = code.split('\n')
    if lineno - 1 >= len(lines):
        return None

    original_line = lines[lineno - 1]

    # For floor div, need exact match; for + and -, be careful with strings
    if old_str in ('*', '//'):
        new_line = original_line.replace(old_str, new_str, 1)
    else:
        # For + and -, only replace when surrounded by spaces or word chars
        pattern = re.compile(r'(\w)\s*' + re.escape(old_str) + r'\s*(\w)')
        m = pattern.search(original_line)
        if m:
            new_line = original_line[:m.start()] + m.group(1) + f' {new_str} ' + m.group(2) + original_line[m.end():]
        else:
            return None

    if new_line == original_line:
        return None

    lines[lineno - 1] = new_line
    return '\n'.join(lines), BugRecord(
        description=f"Arithmetic operator swapped on line {lineno}: '{old_str}' changed to '{new_str}'",
        lines=[lineno],
        fix=fix,
        bug_type="ast_binop_swap",
    )


def ast_boolop_flip_injector(code: str, rng: random.Random) -> Optional[Tuple[str, BugRecord]]:
    """AST-based: find boolean operations and flip And ↔ Or."""
    if not _is_python(code):
        return None

    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return None

    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp):
            candidates.append((node.lineno, type(node.op)))

    if not candidates:
        return None

    lineno, op_type = rng.choice(candidates)
    lines = code.split('\n')
    if lineno - 1 >= len(lines):
        return None

    original_line = lines[lineno - 1]

    if op_type == ast.And:
        new_line = original_line.replace(' and ', ' or ', 1)
        fix = "Change 'or' back to 'and'"
        desc = "'and' changed to 'or'"
    else:
        new_line = original_line.replace(' or ', ' and ', 1)
        fix = "Change 'and' back to 'or'"
        desc = "'or' changed to 'and'"

    if new_line == original_line:
        return None

    lines[lineno - 1] = new_line
    return '\n'.join(lines), BugRecord(
        description=f"Boolean operator flipped on line {lineno}: {desc}",
        lines=[lineno],
        fix=fix,
        bug_type="ast_boolop_flip",
    )


def ast_return_negate_injector(code: str, rng: random.Random) -> Optional[Tuple[str, BugRecord]]:
    """AST-based: find Return statements with boolean/numeric constants and negate them."""
    if not _is_python(code):
        return None

    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return None

    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            if isinstance(node.value, ast.Constant):
                val = node.value.value
                if val is True or val is False:
                    candidates.append((node.lineno, val, 'bool'))
                elif isinstance(val, int) and val in (0, 1, -1):
                    candidates.append((node.lineno, val, 'int'))

    if not candidates:
        return None

    lineno, val, vtype = rng.choice(candidates)
    lines = code.split('\n')
    if lineno - 1 >= len(lines):
        return None

    original_line = lines[lineno - 1]

    if vtype == 'bool':
        if val is True:
            new_line = original_line.replace('True', 'False', 1)
            fix = "Change 'False' back to 'True'"
        else:
            new_line = original_line.replace('False', 'True', 1)
            fix = "Change 'True' back to 'False'"
    else:
        negated = {0: -1, 1: -1, -1: 1}
        new_val = negated.get(val, -val)
        new_line = original_line.replace(f'return {val}', f'return {new_val}', 1)
        fix = f"Change 'return {new_val}' back to 'return {val}'"

    if new_line == original_line:
        return None

    lines[lineno - 1] = new_line
    return '\n'.join(lines), BugRecord(
        description=f"Return value negated on line {lineno}: returns wrong value",
        lines=[lineno],
        fix=fix,
        bug_type="ast_return_negate",
    )


# ─── Injector Registry ──────────────────────────────────────────────────────

# Regex-based injectors (work on all languages)
REGEX_INJECTORS: List[Callable] = [
    off_by_one_injector,
    null_deref_injector,
    wrong_operator_injector,
    unused_var_injector,
    logic_inversion_injector,
]

# AST-based injectors (Python only, more precise)
AST_INJECTORS: List[Callable] = [
    ast_comparison_flip_injector,
    ast_binop_swap_injector,
    ast_boolop_flip_injector,
    ast_return_negate_injector,
]

# Combined list — AST injectors first (preferred for Python)
BUG_INJECTORS: List[Callable] = AST_INJECTORS + REGEX_INJECTORS

INJECTOR_NAMES: Dict[str, Callable] = {
    "off_by_one": off_by_one_injector,
    "null_deref": null_deref_injector,
    "wrong_operator": wrong_operator_injector,
    "unused_var": unused_var_injector,
    "logic_inversion": logic_inversion_injector,
    "ast_comparison_flip": ast_comparison_flip_injector,
    "ast_binop_swap": ast_binop_swap_injector,
    "ast_boolop_flip": ast_boolop_flip_injector,
    "ast_return_negate": ast_return_negate_injector,
}


# ─── Episode Generator ──────────────────────────────────────────────────────

def generate_episode(
    seed: Optional[int] = None,
    difficulty: str = "easy",
) -> Tuple[Snippet, str, List[BugRecord]]:
    """Generate one episode: pick snippet, inject bugs, return gold.

    Args:
        seed: random seed for reproducibility
        difficulty: easy (1 bug), medium (1-2 bugs), hard (2-3 bugs)

    Returns:
        (snippet, buggy_code, gold_bugs)
    """
    rng = random.Random(seed)

    # Filter snippets by difficulty
    pool = [s for s in SNIPPET_BANK if s.difficulty == difficulty]
    if not pool:
        pool = list(SNIPPET_BANK)

    snippet = rng.choice(pool)
    code = snippet.code

    # Determine number of bugs by difficulty
    n_bugs = {"easy": 1, "medium": rng.randint(1, 2), "hard": rng.randint(2, 3)}.get(
        difficulty, 1
    )

    # Apply injectors — prefer AST-based for Python, regex for others
    gold_bugs: List[BugRecord] = []
    if snippet.language == "python":
        # AST injectors first (structurally precise), then regex fallback
        available = list(AST_INJECTORS)
        rng.shuffle(available)
        available += list(REGEX_INJECTORS)
        rng.shuffle(available[len(AST_INJECTORS):])  # shuffle regex portion
    else:
        available = list(REGEX_INJECTORS)
        rng.shuffle(available)

    used_types = set()
    for injector in available:
        if len(gold_bugs) >= n_bugs:
            break
        result = injector(code, rng)
        if result is not None:
            new_code, bug_record = result
            # Avoid duplicate bug types in same episode
            if bug_record.bug_type not in used_types:
                code = new_code
                gold_bugs.append(bug_record)
                used_types.add(bug_record.bug_type)

    # If no bugs were successfully injected, force at least one
    if not gold_bugs:
        for injector in BUG_INJECTORS:
            result = injector(snippet.code, rng)
            if result is not None:
                code, bug_record = result
                gold_bugs.append(bug_record)
                break

    return snippet, code, gold_bugs
