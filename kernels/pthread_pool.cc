#include "pthread_pool.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

struct pool_queue {
	void *arg;
	char free;
	struct pool_queue *next;
};

struct pool {
	char cancelled;
	void *(*fn)(void *);
	unsigned int remaining;
	unsigned int nthreads;
	struct pool_queue *q;
	struct pool_queue *end;
	pthread_mutex_t q_mtx;
	pthread_cond_t q_cnd;
	pthread_t threads[1];
};

static void * thread(void *arg);

void * pool_start(void * (*thread_func)(void *), unsigned int threads) {
	struct pool *p = (struct pool *) malloc(sizeof(struct pool) + (threads-1) * sizeof(pthread_t));
	int i;

	pthread_mutex_init(&p->q_mtx, NULL);
	pthread_cond_init(&p->q_cnd, NULL);
	p->nthreads = threads;
	p->fn = thread_func;
	p->cancelled = 0;
	p->remaining = 0;
	p->end = NULL;
	p->q = NULL;

	for (i = 0; i < threads; i++) {
		pthread_create(&p->threads[i], NULL, &thread, p);
	}

	return p;
}

void pool_enqueue(void *pool, void *arg, char free) {
	struct pool *p = (struct pool *) pool;
	struct pool_queue *q = (struct pool_queue *) malloc(sizeof(struct pool_queue));
	q->arg = arg;
	q->next = NULL;
	q->free = free;

	pthread_mutex_lock(&p->q_mtx);
	if (p->end != NULL) p->end->next = q;
	if (p->q == NULL) p->q = q;
	p->end = q;
	p->remaining++;
	pthread_cond_signal(&p->q_cnd);
	pthread_mutex_unlock(&p->q_mtx);
}

void pool_wait(void *pool) {
	struct pool *p = (struct pool *) pool;

	pthread_mutex_lock(&p->q_mtx);
	while (!p->cancelled && p->remaining) {
		pthread_cond_wait(&p->q_cnd, &p->q_mtx);
	}
	pthread_mutex_unlock(&p->q_mtx);
}

void pool_end(void *pool) {
	struct pool *p = (struct pool *) pool;
	struct pool_queue *q;
	int i;

	p->cancelled = 1;

	pthread_mutex_lock(&p->q_mtx);
	pthread_cond_broadcast(&p->q_cnd);
	pthread_mutex_unlock(&p->q_mtx);

	for (i = 0; i < p->nthreads; i++) {
		pthread_join(p->threads[i], NULL);
	}

	while (p->q != NULL) {
		q = p->q;
		p->q = q->next;

		if (q->free) free(q->arg);
		free(q);
	}

	free(p);
}

static void * thread(void *arg) {
	struct pool_queue *q;
	struct pool *p = (struct pool *) arg;

	while (!p->cancelled) {
		pthread_mutex_lock(&p->q_mtx);
		while (!p->cancelled && p->q == NULL) {
			pthread_cond_wait(&p->q_cnd, &p->q_mtx);
		}
		if (p->cancelled) {
			pthread_mutex_unlock(&p->q_mtx);
			return NULL;
		}
		q = p->q;
		p->q = q->next;
		p->end = (q == p->end ? NULL : p->end);
		pthread_mutex_unlock(&p->q_mtx);

		p->fn(q->arg);

		if (q->free) free(q->arg);
		free(q);
		q = NULL;

		pthread_mutex_lock(&p->q_mtx);
		p->remaining--;
		pthread_cond_broadcast(&p->q_cnd);
		pthread_mutex_unlock(&p->q_mtx);
	}

	return NULL;
}
