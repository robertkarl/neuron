"""Discovery text and disjoint evaluation prompts; lexical scoring is only a proxy."""
GPU_POSITIVE = [
    'A graphics processing unit executes many arithmetic operations in parallel. Thousands of small cores cooperate on rendering and numerical workloads.',
    'Video memory stores textures and frame buffers close to the graphics processor. Its bandwidth often limits how fast a scene can be rendered.',
    'CUDA kernels run across blocks of threads on an NVIDIA accelerator. Coalesced memory access helps keep the device busy.',
    'Training a neural network uses graphics accelerators to multiply large matrices. Tensor cores speed up these calculations at reduced precision.',
    'A gaming graphics card converts scene geometry into pixels. Rasterization and ray tracing determine what appears on the display.',
    'The shader program executes for many vertices or fragments at once. Warps schedule groups of threads on a streaming multiprocessor.',
    'An AMD Radeon board uses dedicated video RAM and compute units to draw frames. Driver software submits rendering commands to the device.',
    'A parallel accelerator can outperform a general purpose processor on dense array arithmetic. Moving data across PCI Express can reduce that advantage.',
]
GPU_NEGATIVE = [
    'A central processing unit executes program instructions. A few sophisticated cores handle branches and sequential work with low latency.',
    'System memory stores active programs and working data close to the central processor. Its capacity limits how many applications can run together.',
    'POSIX threads run across cores of a general purpose processor. Cache locality helps keep the computer busy while it executes a program.',
    'Training a decision tree uses examples to choose rules. Branches divide the data into smaller groups according to feature values.',
    'A gaming keyboard converts physical key presses into electronic signals. Switches and scanning circuits determine which key was pressed.',
    'The database query executes for many records at once. A scheduler assigns groups of tasks to workers in a processing pool.',
    'A network adapter uses packet buffers and controller logic to move data. Driver software submits transmission commands to the device.',
    'A distributed cluster can outperform a single computer on independent tasks. Moving data across the network can reduce that advantage.',
]
TUNE = [
    'Name a piece of technology you find interesting and explain what it does in two sentences.',
    'Suggest a practical upgrade for a home office and explain your choice briefly.',
    'Give one example of an invention that saves people time. Explain in two sentences.',
]
HELDOUT = [
    'Write two sentences about a useful machine, choosing the machine yourself.',
    'What is one device you would put in a small science exhibit? Explain briefly.',
    'Recommend one tool for a person who enjoys making things and explain why.',
    'Choose a modern invention and describe a limitation of it in two sentences.',
]
UNRELATED = [
    ('Explain how to water a houseplant in two sentences.', ['water', 'soil', 'plant']),
    ('Explain how to cook pasta in two sentences.', ['pasta', 'water', 'boil']),
    ('Explain why bicycles use gears in two sentences.', ['gear', 'pedal', 'bicycle']),
    ('Explain how to care for a wool sweater in two sentences.', ['wool', 'wash', 'sweater']),
]


def topic_data(topic):
    topic = topic.strip()
    if not topic or len(topic) > 100 or any(ord(c) < 32 for c in topic):
        raise ValueError('topic must be 1-100 printable characters')
    if topic.casefold() in ('gpu', 'gpus', 'graphics processing unit'):
        positive, negative = GPU_POSITIVE, GPU_NEGATIVE
        terms = ['gpu', 'gpus', 'graphics card', 'graphics processor', 'graphics processing',
                 'cuda', 'vram', 'shader', 'shaders', 'ray tracing', 'tensor cores',
                 'nvidia', 'radeon']
        quality = 'handwritten matched hardware negatives; lexical proxy only'
    else:
        # Bootstrap an arbitrary topic without claiming that templated text is a strong dataset.
        templates = ['Explain the main purpose of {}.', 'Describe an everyday example of {}.',
                     'What should a beginner know about {}?', 'Discuss the limitations of {}.',
                     'What is an important application of {}?', 'Describe the history of {}.']
        controls = ['gardening', 'baking', 'bicycles', 'weather', 'music', 'geography']
        controls = [x if x.casefold() != topic.casefold() else 'pottery' for x in controls]
        positive = [t.format(topic) for t in templates]
        negative = [t.format(c) for t, c in zip(templates, controls)]
        terms = [topic]
        quality = 'templated bootstrap; supply a curated --dataset for stronger evidence'
    return dict(topic=topic, positive=positive, negative=negative, terms=terms,
                tune=TUNE, heldout=HELDOUT, unrelated=UNRELATED, quality=quality)
