import { z, defineCollection } from "astro:content";

const blogSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
    badge: z.string().optional(),
    tags: z.array(z.string()).refine(items => new Set(items).size === items.length, {
        message: 'tags must be unique',
    }).optional(),
});

const eventsSchema = z.object({
    title: z.string(),
    description: z.string(),
    custom_link_label: z.string(),
    custom_link: z.string().optional(),
    updatedDate: z.coerce.date(),
    date: z.string().optional(),
    badge: z.string().optional(),
    checkoutUrl: z.string().optional(),
    heroImage: z.string().optional(),
    tags: z.array(z.string()).refine(items => new Set(items).size === items.length, {
        message: 'tags must be unique',
    }).optional(),
});

const projectsSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
    badge: z.string().optional(),
    tags: z.array(z.string()).refine(items => new Set(items).size === items.length, {
        message: 'tags must be unique',
    }).optional(),
    url: z.string().optional(),
    featured: z.boolean().optional(),
    category: z.string().optional(),
});

const publicationsSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
    badge: z.string().optional(),
    tags: z.array(z.string()).refine(items => new Set(items).size === items.length, {
        message: 'tags must be unique',
    }).optional(),
    journal: z.string().optional(),
    authors: z.array(z.string()).optional(),
    doi: z.string().optional(),
    url: z.string().optional(),
    featured: z.boolean().optional(),
    category: z.string().optional(),
});

export type BlogSchema = z.infer<typeof blogSchema>;
export type EventsSchema = z.infer<typeof eventsSchema>;
export type ProjectsSchema = z.infer<typeof projectsSchema>;
export type PublicationsSchema = z.infer<typeof publicationsSchema>;

const blogCollection = defineCollection({ schema: blogSchema });
const eventsCollection = defineCollection({ schema: eventsSchema });
const projectsCollection = defineCollection({ schema: projectsSchema });
const publicationsCollection = defineCollection({ schema: publicationsSchema });

export const collections = {
    'blog': blogCollection,
    'events': eventsCollection,
    'projects': projectsCollection,
    'publications': publicationsCollection
}